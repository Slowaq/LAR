from robolab_turtlebot import Turtlebot
from .segmentation import find_pylon, find_purple_quads
from .math_utils import *
import numpy as np
import cv2
import math
import rospy
from pprint import pprint
from typing import Tuple, List

EXIT_ANGULAR_VELOCITY = 0.3
DISTANCE_TOL = 0.085
SPEED_TO_THE_POINT = 0.2
ANGULAR_TO_THE_POINT = 0.9
ANGULAR_TO_THE_POINT_CLAMP = 0.7
MINIMAL_ANGULAR_VELOCITY = 0.2
KP_ANG = 5.0   # proportional gain for heading correction, proportional to angle error in radians
KP_ANG_PIXELS = 0.003 # proportional gain for heading correction, proportional to angle error pixels
DISTANCE_OUT_OF_GARAGE = 0.5 # [m] how far should the robot drive out ouf the garade in the exit_garage() method
GARAGE_WALL_DISTANCE = 0.33 # [m] distance from the wall when parking into garage
FREE_SPACE_DISTANCE_THRESHOLD = 0.50
MINIMAL_GARAGE_GATE_ANGULAR_DISTANCE = 0.75 # [rad]
CAMERA_ANGULAR_OFFSET = 0.2 # [rad]
LINEAR_PARKING_VELOCITY = 0.05
PATH_AROUND_PYLON = [(0.0,  0.35), (0.7, 0.35), (0.7, -0.35), (0.0, -0.35)]
SEARCH_FOR_PYLON_PATH = [(0.0, 0.0), (1.5, 0), (3.0, 0)]

class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 
        self.is_running : bool = False  # Blocks B0 from starting another run
        self.record_trajectory : bool = False   # If true, the robot's trajectory will be stored in self.trajectory 
        self.trajectory = []   # Used for storing the trajectory of the robot for debugging purposes. Not used in the algorithm itself.

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        self.stop = False
        self.is_running = True
        self.exit_garage()        
        self.approach_pylon()
        self.drive_around_pylon()
        self.return_to_garage()

        if self.stop:
            print("Algorithm exited early")
        else:
            self.robot.play_sound()
            print("Algorithm successfully finished")
        self.is_running = False

    def exit_garage(self) -> bool:
        """
        The robot orients itself and exits the garage.
        Resets odometry outside of garage.
        """
        if self.find_exit():
            self.drive_out_of_garage()
            self.robot.reset_odometry()
            return True
        else:
            print("Could not find the garage exit!")
            return False

    def find_exit(self) -> bool:
        """
        Rotate robot to the center of the garage's exit using point cloud data.

        The function analyzes the depth point cloud and estimates the
        distance to obstacles in front of the robot. Calculates the angle of the exit 
        by detecting when the wall ends and when it appears again.

        Returns
        -------
        bool
            True if a suitable exit direction was found.
            False if the ROS node shuts down before detection.
        """


        first_wall_end_yaw = None
        second_wall_yaw = None
        found_exit_roughly = False    

        print('Waiting for point cloud ...')
        self.robot.wait_for_point_cloud()
        print('First point cloud recieved ...')

        print("Finding exit")
        while not self.robot.is_shutting_down():
            if self.stop:   
                self.robot.cmd_velocity(0, 0)
                return False

            # get point cloud
            pc = self.robot.get_point_cloud()

            if pc is None:
                print('No point cloud')
                continue
            
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
                self.robot.cmd_velocity(0, MINIMAL_ANGULAR_VELOCITY) # fallback if pointcloud data are horrible
                continue

            current_odom = self.robot.get_odometry()
            if current_odom is None:
                print("Odometry is None")
                continue
            current_yaw = current_odom[2]
            print(f"dist={dist:.2f}, yaw={current_yaw:.3f}")

            # [1] - find exit approximetly
            if not found_exit_roughly:
                self.robot.cmd_velocity(0, EXIT_ANGULAR_VELOCITY)
                if dist >= FREE_SPACE_DISTANCE_THRESHOLD:
                    print("Found the exit roughly")
                    found_exit_roughly = True

            # [2] - we dont even have the first angle
            elif first_wall_end_yaw is None:
                self.robot.cmd_velocity(0, EXIT_ANGULAR_VELOCITY)
                if dist <= FREE_SPACE_DISTANCE_THRESHOLD:
                    first_wall_end_yaw = current_yaw
                    print(f"First wall found at yaw={first_wall_end_yaw:.2f}")
                

            # [2] - we have the first yaw, but not the second one
            elif second_wall_yaw is None:
                self.robot.cmd_velocity(0, -EXIT_ANGULAR_VELOCITY) # rotate counterclockwise
                # Check that we turned far enough away from first edge               
                if dist <= FREE_SPACE_DISTANCE_THRESHOLD and abs(normalize_angle(first_wall_end_yaw - current_yaw)) > MINIMAL_GARAGE_GATE_ANGULAR_DISTANCE:
                    second_wall_yaw = current_yaw
                    print(f"Second wall found at yaw={second_wall_yaw:.2f}")

            # [3] - we have both angles, rotate towards the exit
            else:
                mid_yaw = normalize_angle(
                    (first_wall_end_yaw + second_wall_yaw) / 2 
                )

                if first_wall_end_yaw < second_wall_yaw:
                    mid_yaw += math.pi

                delta_to_mid = normalize_angle(mid_yaw - second_wall_yaw + CAMERA_ANGULAR_OFFSET) # To compensate for the fact that the camera does not head straight ahead 

                print(f"Rotating towards middle of exit: {mid_yaw:.2f}")

                if not self._rotate_by_angle(delta_to_mid):
                    return False

                print("Exit found!")
                return True

        return False

    def drive_out_of_garage(self) -> None:
        """
        The robot drives straight out a short distance in front of the garage.
        Fixed distance.
        
        Returns
        -------
            None
        """
        self._drive_forward(DISTANCE_OUT_OF_GARAGE)
        
    def approach_pylon(self) -> None:
        for point in SEARCH_FOR_PYLON_PATH:
            self._go_to_point_using_odometry(*point)

            if self.look_for_pylon():
                return True
            else:
                print("Couldnt find pylon from this position - trying different point")
        print("Couldnt find pylon at all")
        return False


    def look_for_pylon(self) -> bool:
        """
        Finds the ball and drives to it to a certain distance.
        """
        self._wait_for_new_data()

        initial_yaw = self.robot.get_odometry()[2]
        left_origin = False

        TARGET_DISTANCE = 0.6 

        while not self.robot.is_shutting_down() and not self.stop:
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

            current_yaw = self.robot.get_odometry()[2]
            if abs(normalize_angle(initial_yaw - current_yaw)) > 0.5:
                left_origin = True

            if left_origin and abs(normalize_angle(initial_yaw - current_yaw)) < 0.2:
                print("Robot did full circle and couldnt find pylon")
                cv2.destroyAllWindows()
                return False

            image = np.zeros(pc.shape[:2])

            mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

            image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

            linear = 0.0
            angular = 0.0
            distance = None
                    
            pylon, frame, bw_mask = find_pylon(frame)
                    
            if pylon is None:
                angular = 0.3  # hľadanie objektu
            else:
                column, row = pylon[0], pylon[1]
                pylon_pc = get_average_of_nearby_pixels(pc, row, column)
                if pylon_pc is None:
                    angular = 0.3
                else:
                    x_pixel_error = column - 320                       
                    distance = pylon_pc[2]     
                    angular = -x_pixel_error * 0.001
                    angular = min(max(angular, -0.3), 0.3)
                

            # ak máme validnú vzdialenosť
            if distance is not None and not np.isnan(distance):
                # riadenie dopredného pohybu    
                print(f"distance: {distance:.2f}, x_pixel_error: {x_pixel_error:.2f}, target_distace: {TARGET_DISTANCE:.2f}")
                if distance > TARGET_DISTANCE:
                    print("Going after target")
                    linear = 0.1
                else:
                    print("Close enough to the target - confirming the pylon")
                    self.robot.cmd_velocity(0, 0)   # Wait for accurate reading
                    self._wait_for_new_data()
                    rgb_image = self.robot.get_rgb_image()
                    pc = self.robot.get_point_cloud()
                    pylon, _, _ = find_pylon(rgb_image)

                    if pylon is not None:
                        column, row = pylon
                        pylon_pc = get_average_of_nearby_pixels(pc, row, column)
                        if pylon_pc is not None:
                            # mask_bgr = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)
                            # combined = np.hstack((frame, mask_bgr))
                            # while True:
                            #     cv2.imshow("combined", combined)
                            #     key = cv2.waitKey(1)
                            #     if key == 27:
                            #         break
                            break
                    else:
                        print("ignoring halucination")
                        angular = 0.3
                        
                cv2.putText(frame, f"dist: {distance:.2f} m",
                                    (column - 40, row - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255,255,255), 1)
                
            # Nemame validni vzdalenost - tocime se na miste a hledame pylon
            else:
                pass
                print("Distance is None - searching for pylon")

            print(f"linear={linear:.2f}, angular={angular:.2f}\n")
            self.robot.cmd_velocity(linear=linear, angular=angular)

            mask_bgr = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_bgr))
            cv2.imshow("combined", combined)
            cv2.waitKey(1)

        # cv2.destroyAllWindows()
        distance = pylon_pc[2]
        # We can drive a bit more forward, but the camera wont see the pylon anymore
        self._drive_forward(distance - 0.25) 
        print("Robot successfully found and approached pylon")
        return True


    def drive_around_pylon(self) -> bool:
        """
        Hardcoded maneuver to drive around the pylon using odometry feedback. 
        The robot drives in a rectangle around the pylon and then returns to the starting point.

        Starting point: 28cm in front of the pylon, centered.

        Returns
        -------
            bool: True if successfully drove around the pylon, False if interrupted or failed. 
        """
        print("Driving around pylon using odometry")

        self._wait_for_odometry()
        start_odom = self.robot.get_odometry()

        start_x, start_y, start_phi = start_odom

        # Convert to global frame
        points_global = []
        for x, y in PATH_AROUND_PYLON:
            x_transformed = start_x + x * math.cos(start_phi) - y * math.sin(start_phi)
            y_transformed = start_y + x * math.sin(start_phi) + y * math.cos(start_phi)
            points_global.append((x_transformed, y_transformed))

        # Execute path
        for x_transformed, y_transformed in points_global:
            if not self._go_to_point_using_odometry(x_transformed, y_transformed):
                return False

        return True
    
    def return_to_garage(self) -> bool:
        """Executes the complete sequence to park the robot in the garage.
        
        The robot sequentially approaches the garage, locates the entrance, 
        and drives inside to park.

        Returns:
            bool: True if the entire parking sequence is successful, False otherwise.
        """
        print("returning to garage")
        if not self.approach_garage():
            print("Failed to approach garage")
            return False
        if not self.find_garage_entrance():
            print("Couldnt find garage entrance")
            return False
        if not self.drive_into_garage():
            print("Failed to park into garage")
            return False
        return True

    def find_garage_pillars(self) -> List[Tuple[float, float, float]]:
        """Spins the robot to scan for and locate the two purple garage pillars.
        
        The robot performs a 360-degree rotation, using RGB and point cloud data 
        to identify the pillars. When a pillar is found, the robot stops to get 
        an accurate reading, calculates its global coordinates, and then resumes spinning.

        Returns:
            List[Tuple[float, float, float]]: A list containing the coordinates and 
            yaw of the found pillars in the format (global_x, global_y, center_yaw).
            Returns an empty list if it does not find exactly 2 pillars.
        """
        print('Waiting for point cloud, RGB and odometry...')
        self._wait_for_new_data()
        print('First point cloud, RGB, and odometry received...')

        origin_yaw = self.robot.get_odometry()[2]
        left_origin = False
        found_pillars = []
        stop_spinning = False

        # Do a circle and find purple pillars
        # If the robot sees a purple pillar, it stops moving to get more accurate data
        while not self.robot.is_shutting_down() and not self.stop:
            if not stop_spinning: 
                self.robot.cmd_velocity(0, 0.4)
            else:                
                self.robot.cmd_velocity(0, 0)
                self._wait_for_new_data() # Robot should not be moving while waiting for point cloud

            pc = self.robot.get_point_cloud()   
            rgb_image = self.robot.get_rgb_image()
            odometry = self.robot.get_odometry()
            current_yaw = odometry[2]

            if not left_origin and abs(normalize_angle(current_yaw - origin_yaw)) > 0.5:
                left_origin = True
                print("Left origin")

            if left_origin and abs(normalize_angle(current_yaw - origin_yaw)) < 0.2:
                print("Back at origin")
                cv2.destroyAllWindows()
                break

            pillars, annotated_bgr, bw_image = find_purple_quads(rgb_image)

            if not pillars: 
                print(f"No pillars found")
                continue

            # Focus only on the center that is in the middle of the screen, as the depth camera is most accurate there
            pillars.sort(key=lambda x: abs(x[0] - 320))
            center_of_pillar = pillars[0]
            column, row =  center_of_pillar[0], center_of_pillar[1]
            pillar_pc = None

            if abs(column - 320) < 100 or stop_spinning:
                pillar_pc = get_average_of_nearby_pixels(pc, row, column)
                if pillar_pc is None:
                    print("Pillar center point is None in point cloud")
                    continue
                delta_x = pillar_pc[0]
                delta_y = pillar_pc[2]
                delta_yaw = math.atan2(delta_x, delta_y)
                center_yaw = normalize_angle(current_yaw - delta_yaw)       # Minus because of flipped y-axis compared to global system 
                x, y = rotate_vector(delta_x, delta_y, current_yaw)         # x is right of the robot and y is in front, assuming robot heading is yaw = 0
                print(f"Robot position: x_glob={odometry[0]:.3f} y_glob={odometry[1]:.3f}")
                global_x, global_y = local_coords_to_global_coords(delta_x, delta_y, odometry)       # Global x is in front of the robot and global y is to the left
                
                if stop_spinning:
                    # We have an accurate read
                    found_pillars.append((global_x, global_y, center_yaw))
                    print(f"dx={delta_x:.3f}, dy={delta_y:.3f}, dyaw={delta_yaw:.3f}, yaw={center_yaw:.3f}, x={x:.3f}, y={y:.3f}, robot_yaw={current_yaw:.3f}, glob_x={global_x:.3}, glob_y={global_y:.3f}")
                    # --- VISUALIZATION ---   
                    # TODO: Remove this block - debugging visualization only

                    # if 0 <= row < pc.shape[0] and 0 <= column < pc.shape[1]:
                    #     if center_point is not None:
                    #         cv2.putText(annotated_bgr,
                    #                     "XXX",
                    #                     (column, row),
                    #                     cv2.FONT_HERSHEY_SIMPLEX,
                    #                     0.5,
                    #                     (255, 255, 255),
                    #                     1)
                    #         cv2.putText(annotated_bgr,
                    #                     f"column={column}, row={row}, delta_x={center_delta_x:.2f}, delta_y={center_delta_y:.2f}",
                    #                     (20,20),
                    #                     cv2.FONT_HERSHEY_SIMPLEX,
                    #                     0.5,
                    #                     (255, 255, 255),
                    #                     1)
                            
                    # bw_bgr = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
                    # combined_view = np.hstack((annotated_bgr, bw_bgr))

                    # while True:
                    #     cv2.imshow("RGB + threshold mask (DEBUG)", combined_view.copy())
                    #     key = cv2.waitKey(1)
                    #     if key == 27: #esc
                    #         cv2.destroyAllWindows()
                    #         break                   

                    stop_spinning = False

                # Ignore pillars we have already seen
                elif not any([abs((current_yaw - delta_yaw) - x[2]) < 0.3 for x in found_pillars]):
                    stop_spinning = True    # Robot will stop and wait for fresh point cloud and RGB data
                    continue

                else:
                    print(f"Not stopping for this pillar - the closest center is {min([abs(current_yaw - delta_yaw - x[2]) for x in found_pillars]):.2f}")
            else:
                print("Center is not in the middle of the camera")

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
            bool: True if the rotation to the target angle is successful, False otherwise.
        """
        print("Looking for garage entrance")
        
        pillars = self.find_garage_pillars()
        if pillars:

            pillar_1 = pillars[1][:2]
            pillar_2 = pillars[0][:2]

            # Get garage midpoint (everything is already in global coordinate space)
            print(f"left globally: {pillar_1}")
            print(f"right globally: {pillar_2}")

            garage_gate = average_vector(pillar_1, pillar_2)
            garage_gate = extend_vector(garage_gate, DISTANCE_TOL) # go a bit further to compensate for DISTANCE_TOL
            print(f"garage_gate: {garage_gate}")
            self._go_to_point_using_odometry(*garage_gate)
            target_angle = math.atan2(
                pillar_2[0] - pillar_1[0],
                pillar_2[1] - pillar_1[1]
            )

            # Make sure the robot is not facing the opposite direction
            current_yaw = self.robot.get_odometry()[2]
            if abs(normalize_angle(target_angle - current_yaw)) > math.pi / 2:
                target_angle = normalize_angle(target_angle + math.pi)
        else:
            # Failsafe if finding pillars fails
            target_angle = math.pi
            
        print(f"target angle: {target_angle:.3f}")
        return self._rotate_to_angle(target_angle)

    def approach_garage(self) -> bool:
        """Navigates the robot to the approximate front of the garage.
        
        After this function completes, the robot should be in position 
        to begin searching for the entrance and driving straight in.

        Returns:
            bool: True if it successfully reaches the approach point and rotates, False otherwise.
        """
        if not self._go_to_point_using_odometry(0, 0): # Get in front of garage approximately using odometry
            return False
        self._wait_for_odometry()
        return self._rotate_to_angle(math.pi)

    def drive_into_garage(self) -> bool:
        """Drives the robot straight into the garage using point cloud data.
        
        The robot utilizes depth data to move forward until it reaches a 
        specified distance from the back wall. It assumes the robot is already 
        centered between the pillars and facing the wall.

        Returns:
            bool: True if successfully parked, False if interrupted or if it fails.
        """
        print(f"Driving into garage to a distance of {GARAGE_WALL_DISTANCE:.2f} m from the wall.")
        
        # Rotate towards garage
        # We assume the robot is standing on the axis between the purple pillars

        print("Parking into garage")
        self.robot.reset_odometry()
        self._wait_for_new_data()  

        dest_x = 10        # Tell the robot to go straight
        dest_y = 0
        
        while not self.robot.is_shutting_down():
            if self.stop:   
                self.robot.cmd_velocity(0, 0)
                return False

            current = self.robot.get_odometry()
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
                # print(f"distance={dist:.2f}, target={GARAGE_WALL_DISTANCE}")
                if dist < GARAGE_WALL_DISTANCE:
                    self.robot.cmd_velocity(0, 0)
                    return True
            else:
                dist = float("inf")

            # Desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # Heading error
            angle_error = normalize_angle(desired_yaw - yaw)

            # print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
            #     f"distance={distance:.2f}, angle_error={angle_error:.2f}")    

            # Proportional angular correction
            angular = KP_ANG * angle_error
            angular = max(min(angular, ANGULAR_TO_THE_POINT_CLAMP), -ANGULAR_TO_THE_POINT_CLAMP)  # Clamp

            print(f"Distance: {dist:.2f}, thres: {GARAGE_WALL_DISTANCE:.2f}")
            if dist > GARAGE_WALL_DISTANCE:
                self.robot.cmd_velocity(LINEAR_PARKING_VELOCITY, angular)
            else:
                self.robot.cmd_velocity(0, 0)
                print("Parked into garage!")
                return True

        return False

    def _drive_forward(self, distance: float) -> bool:
        """
        Helper method. Drives given distance (in meters) forward using odometry.
        """
        self._wait_for_odometry()
        odometry = self.robot.get_odometry()
        target_point = local_coords_to_global_coords(0, distance, odometry) 
        return self._go_to_point_using_odometry(*target_point)

    
    def _rotate_by_angle(self,target_delta_yaw: float, angular_speed: float = ANGULAR_TO_THE_POINT) -> bool:
        """
        Rotate the robot by a desired angular displacement using odometry feedback.

        This method performs a closed-loop rotation based on the robot's current yaw.
        A proportional controller is used to smoothly approach the target angle while
        reducing speed near the goal. The rotation stops when the angular error is
        within a small tolerance.

        Parameters
        ----------
        target_delta_yaw : float
            Desired change in orientation (yaw) in radians. Positive values correspond
            to counterclockwise rotation, negative values to clockwise rotation.

        angular_speed : float, optional
            Initial angular velocity in radians per second. This value is dynamically
            adjusted by the proportional controller during execution.

        Returns
        -------
        bool
            True if the rotation was successfully completed, False if interrupted
            (e.g., due to shutdown, stop flag, or missing odometry data).
        """
        print(f"Rotaing by {target_delta_yaw:.2f} with angular speed {angular_speed:.2f}")

        self._wait_for_odometry()
        start = self.robot.get_odometry()
        start_yaw = start[2]

        while not self.robot.is_shutting_down() and not self.stop:
            odom = self.robot.get_odometry()
            if self.record_trajectory:
                self.trajectory.append((odom[0], odom[1]))

            dyaw = normalize_angle(odom[2] - start_yaw)

            angle_error = normalize_angle(target_delta_yaw - dyaw)

            if abs(angle_error) < 0.05:  # ~3 degree tolerance
                break

            # slow down near the end of rotation
            angle_error = normalize_angle(target_delta_yaw - dyaw)
            # print(f"start_yaw={start_yaw:.2f}, current_yaw={odom[2]:.2f}, dyaw={dyaw:.2f}, target_dyaw={target_delta_yaw:.2f}, angle_error={angle_error:.2f}")

            angular = KP_ANG * angle_error   # proportional gain
            angular = max(min(angular, ANGULAR_TO_THE_POINT_CLAMP), -ANGULAR_TO_THE_POINT_CLAMP)  # clamp
            if -MINIMAL_ANGULAR_VELOCITY < angular < MINIMAL_ANGULAR_VELOCITY:
                angular = MINIMAL_ANGULAR_VELOCITY if angular > 0 else -MINIMAL_ANGULAR_VELOCITY

            self.robot.cmd_velocity(0, angular)
            angular_speed=angular

        self.robot.cmd_velocity(0, 0)
        return True
    
    def _rotate_to_angle(self, target_yaw: float, angular_speed: float = ANGULAR_TO_THE_POINT) -> bool:
        """
        Rotate the robot to an absolute yaw angle using odometry.

        Parameters
        ----------
        target_yaw : float
            Desired absolute orientation (yaw) in radians.

        angular_speed : float, optional
            Initial angular velocity.

        Returns
        -------
        bool
            True if rotation completed successfully, False otherwise.
        """
        self._wait_for_odometry()
        odom = self.robot.get_odometry()

        if odom is None:
            print("Odometry is None")
            return False

        current_yaw = odom[2]

        # Compute shortest angular difference
        target_delta_yaw = normalize_angle(target_yaw - current_yaw)

        print(f"Rotating to absolute yaw {target_yaw:.2f} (delta: {target_delta_yaw:.2f})")

        return self._rotate_by_angle(target_delta_yaw, angular_speed)

    def _drive_to_the_point(self, dest_x: float, dest_y: float, speed: float = SPEED_TO_THE_POINT) -> bool:
        """
        Drive the robot toward a target 2D point using odometry-based feedback control.

        The robot continuously adjusts its heading using a proportional angular controller
        to stay aligned with the target point while moving forward. If the heading error
        becomes too large, the robot temporarily stops forward motion and rotates in place
        to correct its orientation.

        Parameters
        ----------
        dest_x : float
            Target x-coordinate in the world frame.

        dest_y : float
            Target y-coordinate in the world frame.

        speed : float, optional
            Desired linear velocity in meters per second when the robot is sufficiently
            aligned with the target direction.

        Returns
        -------
        bool
            True if the robot reaches the target within the specified distance tolerance,
            False if interrupted (e.g., stop flag or shutdown signal).
        """
        print(f"Driving straight to point: ({dest_x:.2f}, {dest_y:.2f})")

        while not self.robot.is_shutting_down():
            if self.stop:   
                self.robot.cmd_velocity(0, 0)
                return False

            # wait for odometry
            cv2.waitKey(10)

            current = self.robot.get_odometry()
            if current is None:
                continue

            x, y, yaw = current
            if self.record_trajectory:
                self.trajectory.append((x, y))

            # distance to goal
            distance = get_distance(current, [dest_x, dest_y])

            # desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # heading error
            angle_error = normalize_angle(desired_yaw - yaw)

            # print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
            #     f"distance={distance:.2f}, angle_error={angle_error:.2f}")

            # stop condition
            if distance < DISTANCE_TOL:
                self.robot.cmd_velocity(0, 0)
                return True

            # proportional angular correction
            angular = KP_ANG * angle_error
            angular = max(min(angular, ANGULAR_TO_THE_POINT_CLAMP), -ANGULAR_TO_THE_POINT_CLAMP)  # Clamp

            # optional: slow down when badly misaligned
            linear = speed
            if abs(angle_error) > 0.5:  # ~30 degrees
                linear = 0.0  # rotate in place if very off

            self.robot.cmd_velocity(linear, angular)

        return False

    def _go_to_point_using_odometry(self, dest_x: float, dest_y: float) -> bool:
        """
        Navigate the robot to a target 2D point using a two-phase odometry-based strategy.

        The navigation consists of:
        1. Rotating the robot to face the target point.
        2. Driving toward the target while maintaining alignment.

        This process relies entirely on odometry feedback and uses helper methods
        for rotation and translation. The function exits early if any step fails
        or if execution is interrupted.

        Parameters
        ----------
        dest_x : float
            Target x-coordinate in the world frame.

        dest_y : float
            Target y-coordinate in the world frame.

        Returns
        -------
        bool
            True if the robot successfully reaches the destination, False if any
            step fails or the operation is interrupted.
        """
        print(f"Driving to point: ({dest_x:.2f}, {dest_y:.2f})")
            
        self._wait_for_odometry()
        current_odom = self.robot.get_odometry()
        if current_odom is None:
            print("Odometry is None")
            return False # Shouldnt ever happen

        current_x, current_y = current_odom[0], current_odom[1]
        distance = get_distance((current_x, current_y), (dest_x, dest_y))
        if distance < DISTANCE_TOL:
            print("Already at the point")
            return True

        current_yaw = current_odom[2]

        # Calculate the required angle to face the destination
        target_angle = math.atan2(dest_y - current_y, dest_x - current_x)
        print(f"Target angle: {target_angle}")

        delta_yaw = normalize_angle(target_angle - current_yaw)

        # ROtate towards point
        angular_speed = ANGULAR_TO_THE_POINT if delta_yaw > 0 else -ANGULAR_TO_THE_POINT            
        if not self._rotate_by_angle(delta_yaw, angular_speed=angular_speed):
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
        Same as Turtlebot.wait_for_rgb_image() while also checking the self.stop flag.
        """
        self.robot.rgb_msg = None
        while not (self.robot.has_rgb_image() or rospy.is_shutdown() or self.stop):
            rospy.sleep(0.5)

    def _wait_for_point_cloud(self) -> None:
        """
        Same as Turtlebot.wait_for_point_cloud() while also checking the self.stop flag.
        """
        self.robot.pc_msg = None
        while not (self.robot.has_point_cloud() or rospy.is_shutdown() or self.stop):
            rospy.sleep(0.5)

    def _wait_for_odometry(self) -> None:
        """
        Same as Turtlebot.wait_for_odometry() while also checking the self.stop flag.
        """
        self.robot.odom = None
        while not (self.robot.has_odometry() or rospy.is_shutdown() or self.stop):
            rospy.sleep(0.5)

    def _wait_for_new_data(self) -> None:
        """
        Waits for new set of rgb_image, point cloud and odometry data while also checking the self.stop flag. 
        """
        self.robot.rgb_msg = None
        self.robot.pc_msg = None
        self.robot.odom = None
        has_new_data = False 
        while not (has_new_data or rospy.is_shutdown() or self.stop):
            rospy.sleep(0.5)
            has_new_data = self.robot.has_odometry() and self.robot.has_rgb_image() and self.robot.has_point_cloud()
