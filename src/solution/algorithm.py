from robolab_turtlebot import Turtlebot, sleep
from .segmentation import find_pylon, find_purple_quads
from .math_utils import (
    get_average_of_nearby_pixels,
    get_distance,
    normalize_angle,
    rotate_vector,
    average_vector,
    extend_vector,
    local_coords_to_global_coords,
    global_coords_to_local_coords,
    clamp_speed,
    line_intesects_circle
)
import numpy as np
import math
from typing import Tuple, List, Optional
import networkx as nx

GARAGE_EXIT_ROTATION_SPEED = 0.3
GOAL_DISTANCE_TOLERANCE = 0.085
DEFAULT_DRIVE_SPEED = 0.2
DEFAULT_ROTATION_SPEED = 0.9
MAX_ROTATION_SPEED = 0.7
MIN_ROTATION_SPEED = 0.2
# Proportional gain for heading correction, proportional to angle error (rad)
HEADING_KP = 5.0
# Distance from wall when parking into garage (meters)
GARAGE_WALL_DISTANCE = 0.34
PYLON_AROUND_PATH = [(0.35,  0.0), (0.35, 0.7), (-0.35, 0.7), (-0.35, 0)]
PYLON_SEARCH_RADIUS = 2
PYLON_SEARCH_POINT_COUNT = 6
POINT_IN_FRONT_OF_GARAGE = (0.7, 0)
CORNER_POINTS_AROUND_GARAGE = [(0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)]

class Algorithm:
    def __init__(self) -> None:
        self.robot: Turtlebot = Turtlebot(rgb=True, depth=True, pc=True)
        # When bumper hits or button pressed, the robot stops
        self.stop: bool = True
        self.is_running: bool = False
        # If true, robot trajectory is stored for debugging
        self.record_trajectory: bool = False
        # List to store the trajectory of the robot
        self.trajectory: List[Tuple[float, float]] = []
        self.safe_points: List[Tuple[float, float]] = CORNER_POINTS_AROUND_GARAGE[:]
        self.pylon_position: Optional[Tuple[float, float]] = None

    def run(self) -> None:
        """
        Execute the full parking algorithm pipeline.

        The method resets the robot state, exits the garage, and searches for
        the pylon. It then drives around the pylon and returns the robot to the
        garage.

        Returns:
            None
        """
        self.stop = False
        self.points_visited = []
        # self.exit_garage()
        self.robot.reset_odometry()
        self.approach_pylon()
        self.drive_around_pylon()
        self.return_to_garage()

        if self.stop:
            self.robot.play_sound(1)
            print("Algorithm exited early")
        else:
            self.robot.play_sound(0)
            print("Algorithm successfully finished")
        self.stop = True

    def exit_garage(self) -> None:
        """
        Orient the robot and exit the garage.

        The method attempts to find the exit direction and resets odometry once
        the exit has been detected.

        Returns:
            None
        """
        if self.find_exit():
            self.robot.reset_odometry()
        else:
            print("Could not find the garage exit!")

    def find_exit(self) -> bool:
        """
        Find and align the robot with the garage exit.

        The method rotates while analyzing depth point cloud data to detect an
        open passage between walls. Once the exit direction is determined, it
        rotates toward that heading and stops.

        Returns:
            bool: True if a suitable exit direction was found, False if the
                operation was interrupted or the exit was not located.
        """

        first_wall_end_yaw = None
        second_wall_yaw = None
        found_exit_roughly = False

        print('Waiting for point cloud and odometry...')
        self._wait_for_point_cloud()
        self._wait_for_odometry()
        print('First point cloud and odometry recieved ...')

        EXIT_FREE_SPACE_THRESHOLD = 0.50
        MIN_GARAGE_GATE_ANGLE = 0.75  # [rad]
        EXIT_CAMERA_YAW_OFFSET = 0.2  # [rad]

        print("Finding exit")
        while not self._is_stopping():
            # get point cloud
            pc = self.robot.get_point_cloud()

            y = pc[:, :, 1]
            z = pc[:, :, 2]

            y_safe = np.where(np.isfinite(y), y, np.inf)
            z_safe = np.where(np.isfinite(z), z, np.inf)

            mask = (
                (y_safe < 0.2) &
                (y_safe > -0.2) &
                (z_safe < 3.0)
            )

            data = np.sort(z[mask])

            if data.size > 50:
                dist = np.percentile(data, 10)
            else:
                # Fallback if pointcloud data are horrible
                self.robot.cmd_velocity(0, MIN_ROTATION_SPEED)
                continue

            current_odom = self.robot.get_odometry()
            if current_odom is None:
                continue
            current_yaw = current_odom[2]
            print(f"dist={dist:.2f}, yaw={current_yaw:.3f}")

            # [1] - find exit approximetly
            if not found_exit_roughly:
                self.robot.cmd_velocity(0, GARAGE_EXIT_ROTATION_SPEED)
                if dist >= EXIT_FREE_SPACE_THRESHOLD:
                    print("Found the exit roughly")
                    found_exit_roughly = True

            # [2] - we dont even have the first angle
            elif first_wall_end_yaw is None:
                self.robot.cmd_velocity(0, GARAGE_EXIT_ROTATION_SPEED)
                if dist <= EXIT_FREE_SPACE_THRESHOLD:
                    first_wall_end_yaw = current_yaw
                    print(f"First wall found at yaw={first_wall_end_yaw:.2f}")

            # [2] - we have the first yaw, but not the second one
            elif second_wall_yaw is None:
                self.robot.cmd_velocity(
                    0, -GARAGE_EXIT_ROTATION_SPEED
                )  # rotate counterclockwise
                # Check that we turned far enough away from first edge
                angle_diff = abs(
                    normalize_angle(first_wall_end_yaw - current_yaw)
                )
                if (
                    dist <= EXIT_FREE_SPACE_THRESHOLD
                    and angle_diff > MIN_GARAGE_GATE_ANGLE
                ):
                    second_wall_yaw = current_yaw
                    print(f"Second wall found at yaw={second_wall_yaw:.2f}")

            # [3] - we have both angles, rotate towards the exit
            else:
                mid_yaw = normalize_angle(
                    (first_wall_end_yaw + second_wall_yaw) / 2
                )

                if first_wall_end_yaw < second_wall_yaw:
                    mid_yaw += math.pi

                # Compensate for camera offset not pointing straight ahead
                delta_to_mid = normalize_angle(
                    mid_yaw - second_wall_yaw + EXIT_CAMERA_YAW_OFFSET
                )

                print(f"Rotating towards middle of exit: {mid_yaw:.2f}")

                if not self._rotate_by_angle(delta_to_mid):
                    return False

                print("Exit found!")
                return True

        self.robot.cmd_velocity(0, 0)
        return False

    def approach_pylon(self) -> None:
        """
        Visit predefined search points and attempt to detect the pylon.

        The robot drives sequentially to points on the search path and
        scans for the pylon from each location. If the pylon is found at any
        point, the method returns early.
        method returns early.

        Returns:
            None
        """
        stack_of_points = CORNER_POINTS_AROUND_GARAGE + [POINT_IN_FRONT_OF_GARAGE]
        while stack_of_points:
            point = stack_of_points.pop()
            if not self._go_to_point_using_odometry(*point):
                print(
                    "Driving to point ({:.2f}, {:.2f}) failed".format(
                        point[0], point[1]
                    )
                )
                return
            self.points_visited.append(point)

            if self.look_for_pylon():
                return
            else:
                print(
                    "Couldnt find pylon from this position "
                    "- trying different point"
                )
                if point in CORNER_POINTS_AROUND_GARAGE or point == POINT_IN_FRONT_OF_GARAGE:
                    target_yaw = math.atan2(point[1], point[0])
                    if not self._rotate_to_angle(target_yaw):
                        return
                    print("Cheking if robot can go forward")
                    if self._is_space_in_front_of_robot_clear():
                        print("Going forward since space is clear")
                        odometry = self.robot.get_odometry()
                        new_point = local_coords_to_global_coords(0, 1.0, odometry)
                        stack_of_points.append(new_point)
                        self.safe_points.append(new_point)
                    else:
                        print("Space in front of robot is not clear")
        else:
            print("Couldnt find pylon at all")
            self.stop = True
            

            

    def look_for_pylon(self) -> bool:
        """
        Locate the pylon and approach it using vision and point cloud data.

        The method uses RGB detection to find the pylon, then refines the
        position with point cloud averaging. It drives toward the pylon until
        a target distance is reached.

        Returns:
            bool: True if the pylon was successfully located and approached,
                False if the robot was interrupted or the pylon was not found.
        """
        self._wait_for_new_data()
        PYLON_APPROACH_DISTANCE = 0.8

        odom = self.robot.get_odometry()
        if odom is None:
            return False
        initial_yaw = odom[2]
        initial_point = (odom[0], odom[1])
        left_origin = False

        while not self._is_stopping():
            # --- RGB OBRAZ ---
            frame = self.robot.get_rgb_image()
            if frame is None:
                print("No RGB image")
                continue

            # --- POINT CLOUD ---
            pc = self.robot.get_point_cloud()
            if pc is None:
                print("Pointcloud is None")
                continue

            odometry = self.robot.get_odometry()
            if odometry is None:
                continue
            current_yaw = odometry[2]
            if abs(normalize_angle(initial_yaw - current_yaw)) > 0.5:
                left_origin = True

            if (
                left_origin
                and abs(normalize_angle(initial_yaw - current_yaw)) < 0.2
            ):
                print("Robot did full circle and couldnt find pylon")
                return False

            image = np.zeros(pc.shape[:2])

            mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

            image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

            linear = 0.0
            angular = 0.0
            distance = None

            pylon, frame, _ = find_pylon(frame)

            if pylon is None:
                angular = 0.4  # hľadanie objektu
            else:
                column, row = pylon[0], pylon[1]
                pylon_pc = get_average_of_nearby_pixels(pc, row, column)
                if pylon_pc is None:
                    angular = 0.4
                else:
                    x_pixel_error = column - 320
                    distance = pylon_pc[2]  # pylon local y

                    # Get local coordinates of garage origin
                    origin_local = global_coords_to_local_coords(
                        0, 0, odometry
                    )
                    origin_local_x = origin_local[0]
                    origin_local_y = origin_local[1]

                    angular_avoid_garage = 0.0
                    # 1. Is it in front of the robot? (y > -0.2)
                    # 2. Is it in front of the pylon?
                    #    (origin_local_y < distance)
                    # 3. Is it close enough to care?
                    #    (Within a 1.5m lookahead distance)
                    # 4. Are we on a collision course? (abs(x) < 0.8)
                    if (
                        origin_local_y > -0.2
                        and origin_local_y < distance
                        and get_distance(
                            (origin_local_x, origin_local_y), (0, 0)
                        ) < 1.5
                        and abs(origin_local_x) < 0.8
                    ):

                        # Scale avoidance: max 0.4 when origin centered
                        avoidance_strength = (
                            1 * (0.8 - abs(origin_local_x)) / 0.8
                        )

                        # If origin is to the right (x >= 0), steer left.
                        # If origin is to the left (x < 0), steer right.
                        if origin_local_x >= 0:
                            angular_avoid_garage = avoidance_strength
                        else:
                            angular_avoid_garage = -avoidance_strength

                    angular_pixel_error = -x_pixel_error * 0.001

                    # Avoid angular takes precedence, pushes robot off
                    angular = min(
                        max(angular_pixel_error + angular_avoid_garage, -0.6),
                        0.6
                    )

            # ak máme validnú vzdialenosť
            if distance is not None and not np.isnan(distance):
                if distance > PYLON_APPROACH_DISTANCE:
                    linear = 0.1
                else:
                    self.robot.cmd_velocity(0, 0)   # Wait for accurate reading
                    self._wait_for_new_data()
                    rgb_image = self.robot.get_rgb_image()
                    pc = self.robot.get_point_cloud()
                    pylon, _, _ = find_pylon(rgb_image)

                    if pylon is not None:
                        column, row = pylon
                        pylon_pc = get_average_of_nearby_pixels(
                            pc, row, column
                        )
                        if pylon_pc is not None:
                            odometry = self.robot.get_odometry()
                            if odometry is None:
                                continue  # Only happens if interrupted
                            distance = pylon_pc[2]
                            # We can drive a bit more forward, but
                            # the camera wont see the pylon anymore
                            pylon_local = (pylon_pc[0], distance)
                            target_point_local = extend_vector(
                                pylon_local, -0.3
                            )
                            target_point = local_coords_to_global_coords(
                                *target_point_local, odometry
                            )
                            self.pylon_position = local_coords_to_global_coords(
                                *pylon_local, odometry
                            )

                            if not self._go_to_point_using_odometry(
                                *target_point
                            ):
                                print("Driving closer to pylon failed")
                                return False
                            print(
                                "Robot successfully found and approached pylon"
                            )
                            return True
                    else:
                        angular = 0.4

            # Nemame validni vzdalenost - tocime se na miste a hledame pylon
            else:
                pass

            self.robot.cmd_velocity(linear=linear, angular=angular)

        # End while - robot was interrupted
        return False

    def drive_around_pylon(self) -> None:
        """
        Execute a hardcoded maneuver around the pylon using odometry.

        The robot drives in a rectangular path around the pylon after it has
        been located, then returns to the start of the local maneuver.

        Returns:
            None
        """
        print("Driving around pylon using odometry")

        self._wait_for_odometry()
        start_odom = self.robot.get_odometry()
        if start_odom is None:
            return    # Only happens if robot got interrupted

        # Convert to global frame
        points_global = []
        for x, y in PYLON_AROUND_PATH:
            point_global = local_coords_to_global_coords(
                x, y, start_odom
            )
            points_global.append(point_global)
            self.safe_points.append(point_global)

        # Execute path
        for point_global in points_global:
            if not self._go_to_point_using_odometry(
                *point_global
            ):
                break

    def return_to_garage(self) -> None:
        """Executes the complete sequence to park the robot in the garage.

        The robot sequentially approaches the garage, locates the entrance,
        and drives inside to park.

        Returns:
            None
        """
        print("returning to garage")
        if not self.approach_garage():
            print("Failed to approach garage")
            return
        if not self.find_garage_entrance():
            print("Couldnt find garage entrance")
            return
        if not self.drive_into_garage():
            print("Failed to park into garage")
            return
        return

    def find_garage_pillars(self) -> List[Tuple[float, float, float]]:
        """Scan for and locate the two purple garage pillars.

        The robot performs a 360-degree rotation using RGB and point cloud
        data
        to identify the pillars. When a pillar is found, the robot stops to get
        an accurate reading, calculates its global coordinates, and resumes
        spinning.

        Returns:
            List[Tuple[float, float, float]]: A list containing the coordinates
                and yaw of the found pillars in the format
                (global_x, global_y, center_yaw). Returns an empty list if it
                does not find exactly 2 pillars.
        """
        print('Waiting for point cloud, RGB and odometry...')
        self._wait_for_new_data()
        print('First point cloud, RGB, and odometry received...')

        CAMERA_ANGULAR_OFFSET = 0.1

        odometry = self.robot.get_odometry()
        if odometry is None:
            return []  # robot got interrupted
        last_yaw = odometry[2]
        found_pillars = []
        stop_spinning = False
        total_rotated = 0.0

        # Do a circle and find purple pillars
        # If the robot sees a purple pillar, it stops moving to get
        # more accurate data
        while not self._is_stopping():
            if not stop_spinning:
                self.robot.cmd_velocity(0, 0.4)
            else:
                self.robot.cmd_velocity(0, 0)
                # Robot should not be moving while waiting for data
                self._wait_for_new_data()

            pc = self.robot.get_point_cloud()
            rgb_image = self.robot.get_rgb_image()
            odometry = self.robot.get_odometry()
            if odometry is None:
                continue  # robot got interrupted
            current_yaw = odometry[2]
            total_rotated += normalize_angle(current_yaw - last_yaw)
            last_yaw = current_yaw

            if total_rotated > 2 * math.pi + 0.2:
                break

            pillars, annotated_brg, image_bw = find_purple_quads(rgb_image)

            if not pillars:
                continue

            # Focus on the center (most accurate for depth camera)
            pillars.sort(key=lambda x: abs(x[0] - 320))
            center_of_pillar = pillars[0]
            column, row = center_of_pillar[0], center_of_pillar[1]
            pillar_pc = None

            if abs(column - 320) < 100 or stop_spinning:
                pillar_pc = get_average_of_nearby_pixels(pc, row, column)
                if pillar_pc is None:
                    print("Pillar center point is None in point cloud")
                    return []
                delta_x = pillar_pc[0]
                delta_y = pillar_pc[2]

                delta_x, delta_y = rotate_vector(
                    delta_x, delta_y, CAMERA_ANGULAR_OFFSET
                )

                delta_yaw = math.atan2(delta_x, delta_y)
                # Minus because of flipped y-axis compared to global system
                center_yaw = normalize_angle(current_yaw - delta_yaw)
                # Global x is in front of robot, global y is to the left
                global_x, global_y = local_coords_to_global_coords(
                    delta_x, delta_y, odometry
                )

                if stop_spinning:
                    # We have an accurate read
                    found_pillars.append((global_x, global_y, center_yaw))

                    stop_spinning = False

                elif not any(
                    [abs((current_yaw - delta_yaw) - x[2]) < 0.4
                     for x in found_pillars]
                ):
                    # Wait for fresh point cloud and RGB data
                    stop_spinning = True
                    continue

        if len(found_pillars) == 2:
            return found_pillars
        else:
            print(f"Found {len(found_pillars)} pillars instead of 2")
            return []

    def find_garage_entrance(self) -> bool:
        """Locates the garage entrance and positions the robot to enter.

        The robot calculates the midpoint between the two found pillars,
        drives to that midpoint, and rotates to face into the garage.

        Returns:
            bool: True if the rotation to the target angle is successful,
                False otherwise.
        """
        print("Looking for garage entrance")

        pillars = self.find_garage_pillars()
        if pillars:

            pillar_1 = pillars[1][:2]
            pillar_2 = pillars[0][:2]

            # Get the garage midpoint (everything is already in global
            # coordinate space)
            print(f"left globally: {pillar_1}")
            print(f"right globally: {pillar_2}")

            garage_gate = average_vector(pillar_1, pillar_2)
            print(f"garage_gate: {garage_gate}")

            # Calculate the vector from pillar 1 to pillar 2
            dx = pillar_2[0] - pillar_1[0]
            dy = pillar_2[1] - pillar_1[1]

            # Perpendicular vector components:
            # normal_x = -dy
            # normal_y = dx
            target_angle = math.atan2(dx, -dy)  # atan2(y_comp, x_comp)

            odometry = self.robot.get_odometry()
            if odometry is None:
                return False    # robot got interrupted
            current_yaw = odometry[2]
            # Make sure the robot is not facing the opposite direction
            if abs(normalize_angle(target_angle - current_yaw)) > math.pi / 2:
                target_angle = normalize_angle(target_angle + math.pi)
            print(f"target angle: {target_angle:.3f}")

            if not self._go_to_point_using_odometry(*garage_gate):
                return False

        else:
            # Failsafe if finding pillars fails
            target_angle = math.pi

        return self._rotate_to_angle(target_angle)

    def approach_garage(self) -> bool:
        """Navigates the robot to the approximate front of the garage.

        After this function completes, the robot should be in position
        to begin searching for the entrance and driving straight in.

        Returns:
            bool: True if it successfully reaches the approach point and
                rotates, False otherwise.
        """
        path = self._get_path_to_garage()
        for point in path:
            # Get in front of garage approximately using odometry
            if not self._go_to_point_using_odometry(*point, go_fast=True):
                return False
        return self._rotate_to_angle(math.pi)

    def drive_into_garage(self) -> bool:
        """Drives the robot straight into the garage using point cloud data.

        The robot uses depth data to move forward until it reaches a specified
        distance from the back wall. It assumes the robot is already centered
        between the pillars and facing the wall.

        Returns:
            bool: True if successfully parked, False if interrupted or if it
                fails.
        """
        print(
            f"Driving into garage to a distance of "
            f"{GARAGE_WALL_DISTANCE:.2f} m from the wall."
        )

        # Rotate towards garage
        # We assume the robot is standing on the axis between the
        # purple pillars

        print("Parking into garage")
        self.robot.reset_odometry()
        self._wait_for_new_data()
        GARAGE_PARKING_LINEAR_SPEED = 0.05

        dest_x = 10        # Tell the robot to go straight
        dest_y = 0

        while not self._is_stopping():
            current = self.robot.get_odometry()
            if current is None:
                continue
            pc = self.robot.get_point_cloud()
            if current is None or pc is None:
                continue

            x, y, yaw = current

            # Distance to goal
            # Mask out floor points
            pc_center = pc[200:280, 280:360, :]  # 80x80x3
            mask = pc_center[:, :, 1] < 0.2

            # Mask points that are too far
            mask = np.logical_and(mask, pc_center[:, :, 2] < 3.0)

            # Check obstacle
            mask = np.logical_and(mask, pc_center[:, :, 1] > -0.2)
            data = np.sort(pc_center[:, :, 2][mask])

            # Stop condition
            if data.size > 50:
                dist = np.percentile(data, 10)
            else:
                dist = float("inf")

            # Desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # Heading error
            angle_error = normalize_angle(desired_yaw - yaw)

            # print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
            #     f"distance={distance:.2f}, angle_error={angle_error:.2f}")

            # Proportional angular correction
            angular = 0.5 * angle_error
            angular = max(
                min(angular, MAX_ROTATION_SPEED), -MAX_ROTATION_SPEED
            )  # Clamp

            print(f"Distance: {dist:.2f}, thres: {GARAGE_WALL_DISTANCE:.2f}")
            if dist > GARAGE_WALL_DISTANCE:
                self.robot.cmd_velocity(GARAGE_PARKING_LINEAR_SPEED, angular)
            else:
                self.robot.cmd_velocity(0, 0)
                print("Parked into garage!")
                return True

        self.robot.cmd_velocity(0, 0)
        return False

    def _is_space_in_front_of_robot_clear(self) -> bool:

        self.robot.cmd_velocity(0, 0)
        self._wait_for_point_cloud()

        pc = self.robot.get_point_cloud()
        if pc is None:
            return False  # robot got interrupted

        # mask out floor points and points too high
        mask = pc[:, :, 1] < -0.1
        mask = np.logical_and(mask, pc[:, :, 1] > -0.3)

        # mask point that are not in front of the robot
        mask = np.logical_and(mask, pc[:, :, 0] < 0.3)
        mask = np.logical_and(mask, pc[:, :, 0] > -0.3)

        data = np.sort(pc[:, :, 2][mask])

        if data.size > 50:
            dist = np.percentile(data, 10)
            print(f"Free distance: {dist:.3f} m")
            if dist > 1.3:
                return True    
        else:
            return None

    def _get_path_to_garage(self) -> List[Tuple[float, float]]:
        self._wait_for_odometry()
        odom = self.robot.get_odometry()
        if odom is None:
            return []   # robot got interrupted
        current_point = (odom[0], odom[1])
        self.safe_points.sort(
            key=lambda x: get_distance(x, current_point)
        )
        start_point = self.safe_points[0] # closest point
        target_point = POINT_IN_FRONT_OF_GARAGE

        graph = nx.Graph()
        for i, point_1 in enumerate(self.safe_points + [target_point]):
            rest_of_safe_points = self.safe_points[:i]
            for point_2 in rest_of_safe_points:
                if line_intesects_circle(
                    point_1,
                    point_2,
                    (0,0),   # origin
                    0.69     # safe radius around garage   
                ):
                    continue
                if line_intesects_circle(
                    point_1,
                    point_2,
                    self.pylon_position,
                    0.3     # safe radius around pylon
                ):
                    continue

                distance = get_distance(point_1, point_2)
                graph.add_edge(point_1, point_2, weight=distance)

        # Get the shortest path
        path = nx.dijkstra_path(graph, source=start_point, target=target_point)
        print(f"Calculated path to garage:")
        from pprint import pprint
        pprint(path)
        return path[1:]  # skip the first point since we are already there

    def _drive_forward(self, distance: float) -> bool:
        """
        Helper method. Drives given distance (in meters) forward using
        odometry.

        Returns:
            bool: True if the forward drive completed, False if interrupted.
        """
        self._wait_for_odometry()
        odometry = self.robot.get_odometry()
        if odometry is None:
            return False
        target_point = local_coords_to_global_coords(0, distance, odometry)
        return self._go_to_point_using_odometry(*target_point)

    def _rotate_by_angle(
        self,
        target_delta_yaw: float,
        go_fast: bool = False
    ) -> bool:
        """
        Rotate the robot by a desired angular displacement using
        odometry feedback.

        This method performs a closed-loop rotation based on the robot's
        current yaw. A proportional controller is used to smoothly approach
        the target angle while reducing speed near the goal. The rotation stops
        when the angular error is within a small tolerance.
        within a small tolerance.

        Parameters
        ----------
        target_delta_yaw : float
            Desired change in orientation (yaw) in radians.
            Positive values correspond to counterclockwise rotation, negative
            values to clockwise rotation.

        angular_speed : float, optional
            Initial angular velocity in radians per second. This value is
            dynamically adjusted by the proportional controller during
            execution.

        Returns
        -------
        bool
            True if the rotation was successfully completed, False if
            interrupted (e.g., due to shutdown, stop flag, or missing odometry
            data).
        """
        print(
            f"Rotaing by {target_delta_yaw:.2f} radians "
        )

        self._wait_for_odometry()
        start = self.robot.get_odometry()
        if start is None:
            return False
        start_yaw = start[2]

        while not self._is_stopping():
            odom = self.robot.get_odometry()
            if odom is None:
                return False
            if self.record_trajectory:
                self.trajectory.append((odom[0], odom[1]))

            dyaw = normalize_angle(odom[2] - start_yaw)

            angle_error = normalize_angle(target_delta_yaw - dyaw)

            if abs(angle_error) < 0.05:  # ~3 degree tolerance
                self.robot.cmd_velocity(0, 0)
                return True

            # slow down near the end of rotation
            angle_error = normalize_angle(target_delta_yaw - dyaw)
            # Debug: print rotation progress

            angular = HEADING_KP * angle_error   # proportional gain
            max_angular = MAX_ROTATION_SPEED

            if go_fast:
                angular *= 1.5
                max_angular *= 2

            angular = clamp_speed(
                angular,
                max_speed=max_angular,
                min_speed=MIN_ROTATION_SPEED
            )

            self.robot.cmd_velocity(0, angular)

        self.robot.cmd_velocity(0, 0)
        return False

    def _rotate_to_angle(
        self, target_yaw: float
    ) -> bool:
        """
        Rotate the robot to an absolute yaw angle using odometry.

        Parameters
        ----------
        target_yaw : float
            Desired absolute orientation (yaw) in radians.

        Returns
        -------
        bool
            True if rotation completed successfully, False otherwise.
        """
        self._wait_for_odometry()
        odom = self.robot.get_odometry()
        if odom is None:
            return False

        current_yaw = odom[2]

        # Compute shortest angular difference
        target_delta_yaw = normalize_angle(target_yaw - current_yaw)

        print(
            f"Rotating to absolute yaw {target_yaw:.2f} "
            f"(delta: {target_delta_yaw:.2f})"
        )

        return self._rotate_by_angle(target_delta_yaw)

    def _drive_to_the_point(
        self, dest_x: float, dest_y: float,
        speed: float = DEFAULT_DRIVE_SPEED
    ) -> bool:
        """
        Drive the robot toward a target 2D point using odometry feedback.

        The robot continuously adjusts its heading using a proportional
        angular controller to stay aligned with the target point while moving
        forward. If the heading error becomes too large, the robot temporarily
        stops forward motion and rotates in place to correct its orientation.
        to correct its orientation.

        Parameters
        ----------
        dest_x : float
            Target x-coordinate in the world frame.

        dest_y : float
            Target y-coordinate in the world frame.

        speed : float, optional
            Desired linear velocity in meters per second when the robot is
            sufficiently aligned with the target direction.

        Returns
        -------
        bool
            True if the robot reaches the target within the specified distance
            tolerance, False if interrupted (e.g., stop flag or shutdown
            signal).
        """
        print(f"Driving straight to point: ({dest_x:.2f}, {dest_y:.2f})")

        while not self._is_stopping():
            self._wait_for_odometry()
            current = self.robot.get_odometry()
            if current is None:
                continue

            x, y, yaw = current
            if self.record_trajectory:
                self.trajectory.append((x, y))

            # distance to goal
            distance = get_distance(current, (dest_x, dest_y))

            # desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # heading error
            angle_error = normalize_angle(desired_yaw - yaw)

            # print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
            #     f"distance={distance:.2f}, angle_error={angle_error:.2f}")

            # stop condition
            if distance < GOAL_DISTANCE_TOLERANCE:
                self.robot.cmd_velocity(0, 0)
                return True

            # proportional angular correction
            angular = HEADING_KP * angle_error
            angular = max(
                min(angular, MAX_ROTATION_SPEED), -MAX_ROTATION_SPEED
            )  # Clamp

            # optional: slow down when badly misaligned
            linear = speed
            if abs(angle_error) > 0.5:  # ~30 degrees
                linear = 0.0  # rotate in place if very off

            self.robot.cmd_velocity(linear, angular)

        self.robot.cmd_velocity(0, 0)
        return False

    def _go_to_point_using_odometry(
        self, 
        dest_x: float, 
        dest_y: float,
        go_fast: bool = False
    ) -> bool:
        """
        Navigate the robot to a target 2D point using a two-phase
        odometry-based strategy.

        The navigation consists of:
        1. Rotating the robot to face the target point.
        2. Driving toward the target while maintaining alignment.

        This process relies entirely on odometry feedback and uses helper
        methods for rotation and translation. The function exits early if any
        step fails or if execution is interrupted.

        Parameters
        ----------
        dest_x : float
            Target x-coordinate in the world frame.

        dest_y : float
            Target y-coordinate in the world frame.

        go_fast : bool, optional
            If True, the robot will use a higher linear and angular speed when driving
            toward the target point. Faster, but less accurate. Default is False.

        Returns
        -------
        bool
            True if the robot successfully reaches the destination,
            False if any step fails or the operation is interrupted.
        """
        print(f"Driving to point: ({dest_x:.2f}, {dest_y:.2f})")

        self._wait_for_odometry()
        current_odom = self.robot.get_odometry()
        if current_odom is None:
            return False

        current_x, current_y = current_odom[0], current_odom[1]
        distance = get_distance((current_x, current_y), (dest_x, dest_y))
        if distance < GOAL_DISTANCE_TOLERANCE:
            print("Already at the point")
            return True

        current_yaw = current_odom[2]

        # Calculate the required angle to face the destination
        target_angle = math.atan2(dest_y - current_y, dest_x - current_x)

        delta_yaw = normalize_angle(target_angle - current_yaw)

        if not self._rotate_by_angle(
            delta_yaw, 
            go_fast=go_fast
        ):
            print("Rotating towards point failed.")
            return False
        else:
            print("Successfully rotated towards point.")

        # Drive to the point
        if not self._drive_to_the_point(dest_x, dest_y):
            print("Driving towards point failed.")
            return False
        else:
            print("Successfully drove towards point.")

        print("Destination reached successfully!")
        return True

    def _wait_for_rgb_image(self) -> None:
        """
        Wait for a fresh RGB image while respecting the stop flag.

        Returns:
            None
        """
        self.robot.rgb_msg = None
        while not (self.robot.has_rgb_image() or self._is_stopping()):
            sleep(0.5)

    def _wait_for_point_cloud(self) -> None:
        """
        Wait for a fresh point cloud while respecting the stop flag.

        Returns:
            None
        """
        self.robot.pc_msg = None
        while not (self.robot.has_point_cloud() or self._is_stopping()):
            sleep(0.5)

    def _wait_for_odometry(self) -> None:
        """
        Wait for fresh odometry data while respecting the stop flag.

        Returns:
            None
        """
        self.robot.odom = None
        while not (self.robot.has_odometry() or self._is_stopping()):
            sleep(0.5)

    def _wait_for_new_data(self) -> None:
        """
        Wait for a new set of RGB, point cloud, and odometry data.

        The method clears the cached sensor messages and blocks until all
        three data sources are available or the stop flag becomes true.

        Returns:
            None
        """
        self.robot.rgb_msg = None
        self.robot.pc_msg = None
        self.robot.odom = None
        has_new_data = False
        while not (has_new_data or self._is_stopping()):
            sleep(0.5)
            has_new_data = (
                self.robot.has_odometry()
                and self.robot.has_rgb_image()
                and self.robot.has_point_cloud()
            )

    def _is_stopping(self) -> bool:
        """
        Check whether execution should stop.

        Returns:
            bool: True if ROS is shutting down or the self.stop flag is set.
        """
        return self.robot.is_shutting_down() or self.stop
