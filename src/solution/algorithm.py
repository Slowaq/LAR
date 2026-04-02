from robolab_turtlebot import Turtlebot
from .segmentation import find_pylon, find_purple_quads
from .math_utils import *
import numpy as np
import cv2
import math
from pprint import pprint

EXIT_ANGULAR_VELOCITY = 0.3
DISTANCE_TOL = 0.085
SPEED_TO_THE_POINT = 0.3
ANGULAR_TO_THE_POINT = 0.7
ANGULAR_TO_THE_POINT_CLAMP = 0.5
MINIMAL_ANGULAR_VELOCITY = 0.10
KP_ANG = 5.0   # proportional gain for heading correction, proportional to angle error in radians
KP_ANG_PIXELS = 0.003 # proportional gain for heading correction, proportional to angle error pixels
DISTANCE_OUT_OF_GARAGE = 0.5 # [m] how far should the robot drive out ouf the garade in the exit_garage() method
GARAGE_WALL_DISTANCE = 0.26 # [m] distance from the wall when parking into garage
FREE_SPACE_DISTANCE_THRESHOLD = 0.50
MINIMAL_GARAGE_GATE_ANGULAR_DISTANCE = 0.75 # [rad]
CAMERA_ANGULAR_OFFSET = 0.2 # [rad]
LINEAR_PARKING_VELOCITY = 0.1
PATH_AROUND_PYLON = [(0.0,  0.33), (0.65, 0.33), (0.65, -0.33), (0.0, -0.33)]

class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 
        self.trajectory = []   # Used for storing the trajectory of the robot for debugging purposes. Not used in the algorithm itself. # TODO remove in final code

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        self.stop = False
        # self.exit_garage()
        self.robot.reset_odometry()
        # self.approach_pylon()
        # self.drive_around_pylon()
        self.return_to_garage()

        if self.stop:
            print("Algorithm exited early")
        else:
            print("Algorithm successfully finished")

    def exit_garage(self) -> None:
        """
        The robot orients itself and exits the garage.
        """
        if self.find_exit():
            self.drive_out_of_garage()
        else:
            print("Could not find the garage exit!")

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

        self.robot.reset_odometry()
        self._go_to_point_using_odometry(DISTANCE_OUT_OF_GARAGE, 0)
        

    def approach_pylon(self) -> None:
        """
        Finds the ball and drives to it to a certain distance.
        """
        self.robot.wait_for_rgb_image()



        RESPONSE = 0.003

        TARGET_DISTANCE = 0.6  # 50 cm

        stoping = False


        while not self.robot.is_shutting_down() and not self.stop and not stoping:
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

            image = np.zeros(pc.shape[:2])

            mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

            image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

            depth_vis = cv2.applyColorMap(
                255 - image.astype(np.uint8),
                cv2.COLORMAP_JET
            )     # used only for visualization

            linear = 0.0
            angular = 0.0
                    
            error, x, y, frame = find_pylon(frame)
                    
            if error is None:
                angular = 0.3  # hľadanie objektu
            else:
                angular = -KP_ANG_PIXELS * error
                

            # --- ZÍSKANIE VZDIALENOSTI Z POINT CLOUDU ---

            if x is not None and y is not None: 
                distance = pc[y, x, 2]  # Z = vzdialenosť dopredu, pc.shape = (480, 640, 3)
            else:
                distance = None

            # ak máme validnú vzdialenosť
            if distance is not None and not np.isnan(distance):
                # riadenie dopredného pohybu    
                print(f"distance: {distance:.2f}, target_distace: {TARGET_DISTANCE:.2f}")
                if distance > TARGET_DISTANCE:
                    if abs(error) < 100:
                        print("Going after target")
                        linear = 0.1    # jedem dopredu
                    else:
                        print("Angular error is too large - turning on the spot")
                        linear = 0.0    # jenom otaceni
                else:
                        print("Close enough to the target - stopping")
                        linear = 0.0  # zastav
                        stoping = True

                        # debug info
                cv2.putText(frame, f"dist: {distance:.2f} m",
                                    (x - 40, y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255,255,255), 1)
                
            # Nemame validni vzdalenost - tocime se na miste a hledame pylon
            else:
                pass
                print("Distance is None - searching for pylon")

            print(f"linear={linear:.2f}, angular={angular:.2f}\n")
            self.robot.cmd_velocity(linear=linear, angular=angular)

            combined = np.hstack((frame, depth_vis))
            cv2.imshow("combined", combined)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

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

        self.robot.wait_for_odometry()
        start_odom = self.robot.get_odometry()
        if start_odom is None:
            print("Odometry is None")
            return False

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
    
    def return_to_garage(self) -> None:
        """
        The robot finds the garage door, drives in front of it, and then parks inside the garage.
        """
        print("returning to garage")
        # if not self.approach_garage():
        #     print("Failed to approach garage")
        #     return
        if not self.drive_into_garage():
            print("Failed to park into garage")

    def find_garage_entrance(self) -> None:
        """
        The robot turns towards the garage entrance.
        """
        pass

    def approach_garage(self) -> bool:
        """
        The robot drives in front of the garage door. After this function, it should be enough
        to drive straight into the garage.
        """
        if not self._go_to_point_using_odometry(0.3, 0):
            return False
        self.robot.wait_for_odometry()
        return self._rotate_to_angle(math.pi)

    def drive_into_garage(self) -> bool:
        """
        This method uses point cloud data to drive straight into the garage until it is close enough to the wall.
        Neccesary condittion is that the robot is already in between garage pillars and facing the wall. 

        Returns
        -------
            bool: True if successfully parked, False if interrupted or failed.
        """
        print(f"Driving into garage to a distance of {GARAGE_WALL_DISTANCE:.2f} m from the wall.")

        print('Waiting for point cloud, RGB and odometry...')
        self.robot.wait_for_point_cloud()
        self.robot.wait_for_rgb_image()
        self.robot.wait_for_odometry()
        print('First point cloud, RGB and odometry recieved recieved ...')

        # TODO: najit pilire, brat v potaz to, ze aby hloubka byla presna, tak musi robot
        # byt natocen k piliri primo. Pilir nesmi byt na okraji obrazovky - vznika chyba.
        # Podle piliru dopocitat bod, na ose mezi piliri pred garazi a dojet tam a natocit se presne do garaze 

        left_center_target_yaw, right_center_target_yaw = None, None
        origin_yaw = self.robot.get_odometry()[2]
        left_origin = False
        found_centers_yaw = []
        stop_spinning = False

        # [1] Find the two purple pillars - do a circle
        while not self.robot.is_shutting_down() and not self.stop:
            if not stop_spinning: 
                self.robot.cmd_velocity(0, 0.2)
            else:                
                self.robot.cmd_velocity(0, 0)
                self.robot.wait_for_point_cloud()
                self.robot.wait_for_rgb_image()
                self.robot.cmd_velocity(0,0)

            pc = self.robot.get_point_cloud()   # Robot should not be moving while waiting for point cloud
            rgb_image = self.robot.get_rgb_image()
            odometry = self.robot.get_odometry()
            current_yaw = odometry[2]

            if not left_origin and abs(normalize_angle(current_yaw - origin_yaw)) > 0.5:
                left_origin = True
                print("Left origin")

            if left_origin and abs(normalize_angle(current_yaw -origin_yaw)) < 0.2:
                print("Back at origin")
                pprint(found_centers_yaw)
                left_origin = False
                # cv2.destroyAllWindows()
                # break

            centers, annotated_bgr, _ = find_purple_quads(rgb_image)

            if not centers: 
                print(f"No centers found")
                continue

            # Focus only on the center that is in the middle of screen, because that is where depth camera is the most accurate
            centers.sort(key=lambda x: abs(x[0] - 320))
            center = centers[0]
            column, row =  center[0], center[1]
            center_point = None

            if abs(center[0]-320) < 100:
                if stop_spinning:
                    # Get accurate read
                    center_point = get_average_of_nearby_pixels(pc, row, column)
                    if center_point is None:
                        print("Center point is none in point cloud")
                        continue
                    center_delta_x = center_point[0]
                    center_delta_y = center_point[2]
                    center_delta_yaw = math.atan2(center_delta_x, center_delta_y)
                    center_yaw = normalize_angle(current_yaw - center_delta_yaw)
                    x, y = rotate_vector(center_delta_x, center_delta_y, -center_yaw)       # x is right of the robot and y is in front of the robot, assuming robot is heading at yaw = 0
                    found_centers_yaw.append(center_yaw)
                    pprint(found_centers_yaw)
                    print(f"dx={center_delta_x:.2f}, dy={center_delta_y:.2f}, dyaw={center_delta_yaw:.2f}, yaw={center_yaw:.2f}, x={x:.2f}, y={y:.2f}")
                    print("Starting spinning")
                    stop_spinning = False

                elif not any([abs(center_yaw - x) < 0.2 for x in found_centers_yaw]):
                    print("Stopping spinning")
                    stop_spinning = True    # Robot will stop and wait for fresh pointcloud and rgb data
                    continue

                else:
                    print(f"Not stopping for this - the closes center is {min([abs(center_yaw - x) < 0.2 for x in found_centers_yaw])}")
            else:
                print("Center is not in the middle of camera")


            # --- VISUALIZATION ---   # TODO remove ts - debugging visualisation only

            if 0 <= row < pc.shape[0] and 0 <= column < pc.shape[1]:
                if center_point is not None:
                    distance = center_point[2]
                    if not np.isnan(distance):
                        cv2.putText(annotated_bgr,
                                    f"{distance:.2f} m, column={column:.2f},row={row:.2f}pc={center_point}",
                                    (column - 40, row - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1)

            cv2.imshow("RGB + Depth (DEBUG)", annotated_bgr.copy())
            cv2.waitKey(1)

        return

        # [2] look at the left pillar directly to get the most accurate depth approximation
        if not self._rotate_to_angle(left_center_target_yaw):
            return False
        
        self.robot.wait_for_point_cloud()
        self.robot.wait_for_rgb_image()
        self.robot.wait_for_odometry()
        left_actual_yaw = self.robot.get_odometry()[2]
        rgb_image = self.robot.get_rgb_image()
        pc = self.robot.get_point_cloud()
        centers, _, _ = find_purple_quads(rgb_image)
        if not centers:
            print("Robot does not see left pillar")
            return False
        # get the center closest to center of camera
        centers.sort(key=lambda x: abs(x[0] - 320))
        column, row = centers[0][1], centers[0][0]
        left_pillar = get_average_of_nearby_pixels(pc, column, row)
        if left_pillar is None:
            print("Could not read left pillar")
        # else:
        #     vis = rgb_image.copy()

        #     # draw point
        #     cv2.circle(vis, (row, column), 6, (0, 255, 0), -1)

        #     # format text
        #     text = f"L: ({left_pillar[0]:.2f}, {left_pillar[1]:.2f}, {left_pillar[2]:.2f})"

        #     # draw text near the point
        #     cv2.putText(vis, text, (row + 10, column - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #     while True:
        #         cv2.imshow("Left Pillar", vis)
        #         key = cv2.waitKey(1) 
        #         if key == 27:
        #             cv2.destroyAllWindows()
        #             break

        # [3] look at the right pillar directly to get the most accurate depth approximation
        if not self._rotate_to_angle(right_center_target_yaw):
            return False
        
        self.robot.wait_for_point_cloud()
        self.robot.wait_for_rgb_image()
        self.robot.wait_for_odometry()
        right_actual_yaw = self.robot.get_odometry()[2]
        rgb_image = self.robot.get_rgb_image()
        pc = self.robot.get_point_cloud()
        centers, _, _ = find_purple_quads(rgb_image)
        if not centers:
            print("Robot does not see right pillar")
            return False
        # get the center closest to center of camera
        centers.sort(key=lambda x: abs(x[0] - 320))
        column, row = centers[0][1], centers[0][0]
        right_pillar = get_average_of_nearby_pixels(pc, column, row)
        if right_pillar is None:
            print("Could not read right pillar")
        # else:
        #     vis = rgb_image.copy()

        #     # draw point
        #     cv2.circle(vis, (row, column), 6, (0, 0, 255), -1)

        #     # format text
        #     text = f"R: ({right_pillar[0]:.2f}, {right_pillar[1]:.2f}, {right_pillar[2]:.2f})"

        #     # draw text near the point
        #     cv2.putText(vis, text, (row + 10, column - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #     while True:
        #         cv2.imshow("Right Pillar", vis)
        #         key = cv2.waitKey(1) 
        #         if key == 27:
        #             cv2.destroyAllWindows()
        #             break

        # [4] get garage midpoint (everything is relative to robot
        left = (left_pillar[0], left_pillar[2]) # (x,y), where x is right of robot and y is in front of robot
        print(f"left pillar originaly: {left}")
        right = (right_pillar[0], right_pillar[2])
        print(f"right pillar: {right}")
        phi = normalize_angle(left_actual_yaw - right_actual_yaw)
        left = rotate_vector(*left, phi)  
        print(f"left yaw: {left_actual_yaw:.2f}, right yaw: {right_actual_yaw:.2f}")
        print(f"left pillar after rotation by {phi:.2f}: {left}")
        garage_gate = average_vector(left, right)
        print(f"garage_gate: {garage_gate}")


        # [5] calculate the point to go to
        normal = normalize_vector(substract_vectors(left, right))
        target_point = multiply_vector(
            normal,
            dot_product(normal, garage_gate)
        )

        print(f"Normal: {normal}")
        print(f"Dot product: {dot_product(normal, garage_gate)}")
        print(f"target point (local): {target_point}")
        print(f"current position: {self.robot.get_odometry()}")

        def local_coords_to_global(local_x: float, local_y: float) -> tuple:
            """
            local x is positive right of the robot. 
            local y is positive to the front of the robot.
            Global x is positive in front (North/Forward).
            Global y is positive to the left (West/Left).
            """
            # x, y are the robot's current position in the global frame
            rx, ry, yaw = self.robot.get_odometry()

            # Step 1: Map local inputs to a standard 'Forward/Left' local frame
            # Your local_y is forward (standard local x)
            # Your local_x is right (so -local_x is standard local y)
            forward = local_y
            left = -local_x

            # Step 2: Apply rotation matrix
            # global_x = robot_x + (forward * cos(yaw) - left * sin(yaw))
            # global_y = robot_y + (forward * sin(yaw) + left * cos(yaw))
            
            glob_x = rx + (forward * math.cos(yaw) - left * math.sin(yaw))
            glob_y = ry + (forward * math.sin(yaw) + left * math.cos(yaw))

            return (glob_x, glob_y)

        target_point = local_coords_to_global(*target_point)
        print(f"Target point (global): { target_point}")
        self._go_to_point_using_odometry(*target_point)

        return

        # [6] Rotate towards garage

        # Predpokladame, ze robot stoji na ose mezi fialovymi piliri

        print("Parking into garage")
        self.robot.reset_odometry()
        self.robot.wait_for_odometry()
        self.robot.wait_for_point_cloud()

        dest_x = 10,        # tell the robot to go straight
        dest_y = 0,
        while not self.robot.is_shutting_down():
            if self.stop:   
                self.robot.cmd_velocity(0, 0)
                return False

            current = self.robot.get_odometry()
            pc = self.robot.get_point_cloud()
            if current is None or pc is None:
                continue

            x, y, yaw = current

            # distance to goal

            # mask out floor points
            pc_center = pc[200:280, 280:360, :]  # 80x80x3
            mask = pc_center[:, :, 1] < 0.2

            # mask point too far
            mask = np.logical_and(mask, pc_center[:, :, 2] < 3.0)

            # check obstacle
            mask = np.logical_and(mask, pc_center[:, :, 1] > -0.2)
            data = np.sort(pc_center[:, :, 2][mask])

            # stop condition
            if data.size > 50:
                dist = np.percentile(data, 10)
                # print(f"distance={dist:.2f}, target={GARAGE_WALL_DISTANCE}")
                if dist < GARAGE_WALL_DISTANCE:
                    self.robot.cmd_velocity(0, 0)
                    return True

            # desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # heading error
            angle_error = normalize_angle(desired_yaw - yaw)

            # print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
            #     f"distance={distance:.2f}, angle_error={angle_error:.2f}")    

            # proportional angular correction
            angular = KP_ANG * angle_error
            angular = max(min(angular, ANGULAR_TO_THE_POINT_CLAMP), -ANGULAR_TO_THE_POINT_CLAMP)  # Clamp

            self.robot.cmd_velocity(0.05, angular)


            if dist > GARAGE_WALL_DISTANCE:
                self.robot.cmd_velocity(LINEAR_PARKING_VELOCITY, 0)
            else:
                self.robot.cmd_velocity(0, 0)
                print("Parked into garage!")
                return True

        return False

    def _drive_forward(self) -> None:
        """
        Helper method. Local wrapper around self.robot.cmd_velocity(). Checks the self.stop flag.
        """
        pass

    
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

        self.robot.wait_for_odometry()
        start = self.robot.get_odometry()
        if start is None:
            # Shouldnt ever happen
            print("Odometry is None")
            return False

        start_yaw = start[2]

        while not self.robot.is_shutting_down() and not self.stop:
            odom = self.robot.get_odometry()
            if odom is None:
                # Shouldnt ever happen
                print("Odometry is None")
                continue
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
        self.robot.wait_for_odometry()
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
            
        self.robot.wait_for_odometry()
        current_odom = self.robot.get_odometry()
        if current_odom is None:
            print("Odometry is None")
            return False # Shouldnt ever happen

        current_x, current_y = current_odom[0], current_odom[1]

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