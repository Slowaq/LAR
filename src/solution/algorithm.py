from robolab_turtlebot import Turtlebot
from .segmentation import *
import numpy as np
import cv2
import math

EXIT_ANGULAR_VELOCITY = 0.2
DISTANCE_TOL = 0.085
SPEED_TO_THE_POINT = 0.3
ANGULAR_TO_THE_POINT = 0.7
ANGULAR_TO_THE_POINT_CLAMP = 0.5
MINIMAL_ANGULAR = 0.10
KP_ANG = 5.0   # proportional gain for heading correction
DISTANCE_OUT_OF_GARAGE = 0.5 # [cm] how far should the robot drive out ouf the garade in the exit_garage() method
GARAGE_WALL_DISTANCE = 0.26 # [cm] distance from the wall when parking into garage



class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        self.stop = False
        self.exit_garage()
        self.robot.reset_odometry()
        self.approach_pylon()
        self.drive_around_pylon()
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

        FREE_TH = 0.50
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

            mask = pc[:, :, 1] < 0.2                         # mask out floor points
            mask = np.logical_and(mask, pc[:, :, 2] < 3.0)   # mask point too far
            mask = np.logical_and(mask, pc[:, :, 1] > -0.2)  # check obstacle
            data = np.sort(pc[:, :, 2][mask])

            if data.size > 50:
                dist = np.percentile(data, 10)
            else:
                self.robot.cmd_velocity(0, 0.1) # fallback if pointcloud data are horrible
                continue

            current_odom = self.robot.get_odometry()
            if current_odom is None:
                continue
            current_yaw = current_odom[2]
            print(f"dist={dist:.2f}, yaw={current_yaw:.3f}")

            # [1] - find exit approximetly
            if not found_exit_roughly:
                self.robot.cmd_velocity(0, 0.6)
                if dist >= FREE_TH + 0.05:
                    print("Found the exit roughly")
                    found_exit_roughly = True

            # [2] - we dont even have the first angle
            elif first_wall_end_yaw is None:
                self.robot.cmd_velocity(0, EXIT_ANGULAR_VELOCITY)
                if dist <= FREE_TH:
                    first_wall_end_yaw = current_yaw
                    print(f"First wall found at yaw={first_wall_end_yaw:.2f}")
                

            # [2] - we have the first yaw, but not the second one
            elif second_wall_yaw is None:
                self.robot.cmd_velocity(0, -EXIT_ANGULAR_VELOCITY) # rotate counterclockwise
                # Check that we turned far enough away from first edge               
                if dist <= FREE_TH and abs(self._normalize_angle(first_wall_end_yaw - current_yaw)) > 0.25:
                    second_wall_yaw = current_yaw
                    print(f"Second wall found at yaw={second_wall_yaw:.2f}")

            # [3] - we have both angles, rotate towards the exit
            else:
                mid_yaw = self._normalize_angle(
                    (first_wall_end_yaw + second_wall_yaw) / 2 
                )

                delta_to_mid = self._normalize_angle(mid_yaw - second_wall_yaw + 0.20) # To compensate for the fact that the camera does not head straight ahead 

                print(f"Rotating towards middle of exit: {mid_yaw:.2f}")

                if not self._rotate_towards_point(delta_to_mid):
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

        target_distance = 0.5  # 50 cm
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

            # depth_vis = cv2.applyColorMap(
            #     255 - image.astype(np.uint8),
            #     cv2.COLORMAP_JET
            # )     # used only for visualization

            linear = 0.0
            angular = 0.0
                    
            error, x, y, frame = find_pylon(frame)
                    
            if error is None:
                angular = 0.3  # hľadanie objektu
            else:
                angular = -RESPONSE * error
                

            # --- ZÍSKANIE VZDIALENOSTI Z POINT CLOUDU ---

            if x is not None and y is not None: 
                distance = pc[y, x, 2]  # Z = vzdialenosť dopredu, pc.shape = (480, 640, 3)
            else:
                distance = None

            # ak máme validnú vzdialenosť
            if distance is not None and not np.isnan(distance):
                # riadenie dopredného pohybu    
                print(f"distance: {distance:.2f}, target_distace: {target_distance:.2f}")
                if distance > target_distance:
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

        #     combined = np.hstack((frame, depth_vis))
        #     cv2.imshow("combined", combined)
        #     cv2.waitKey(1)

        # cv2.destroyAllWindows()

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

        # Rectangle in robot frame (forward = x, left = y)
        points_local = [
            (0.0,  0.33),
            (0.75, 0.33),
            (0.75, -0.33),
            (0.0, -0.33),
        ]

        # Convert to global frame
        points_global = []
        for x, y in points_local:
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
        if not self.approach_garage():
            print("Failed to approach garage")
            return
        if not self.drive_into_garage():
            print("Failed to park into garage")

    def find_garage_entrance(self) -> None:
        """
        The robot turns towards the garage entrance.
        """
        pass

    def approach_garage(self) -> None:
        """
        The robot drives in front of the garage door. After this function, it should be enough
        to drive straight into the garage.
        """
        self._go_to_point_using_odometry(0, 0)

    def drive_into_garage(self) -> bool:
        """
        This method uses point cloud data to drive straight into the garage until it is close enough to the wall.
        Neccesary condittion is that the robot is already in between garage pillars and facing the wall. 

        Returns
        -------
            bool: True if successfully parked, False if interrupted or failed.
        """
        print(f"Driving into garage to a distance of {GARAGE_WALL_DISTANCE:.2f} m from the wall.")

        print('Waiting for point cloud ...')
        self.robot.wait_for_point_cloud()
        direction = None
        print('First point cloud recieved ...')

        while not self.robot.is_shutting_down() and not self.stop:
            # get point cloud
            pc = self.robot.get_point_cloud()

            if pc is None:
                print('No point cloud')
                continue

            mask = pc[:, :, 1] < 0.2                         # mask out floor points
            mask = np.logical_and(mask, pc[:, :, 2] < 3.0)   # mask point too far
            mask = np.logical_and(mask, pc[:, :, 1] > -0.2)  # check obstacle
            data = np.sort(pc[:, :, 2][mask])

            if data.size > 50:
                dist = np.percentile(data, 10)
            else:
                self.robot.cmd_velocity(0, 0.1) # fallback if pointcloud data are horrible
                continue

            print(f"dist={dist:.2f}")

            if dist > GARAGE_WALL_DISTANCE:
                self.robot.cmd_velocity(0.1, 0)
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

    def _distance_from(self, point1: list, point2: list) -> float:
        """
        Helper method used for calculating distance of two 2D points.

        Parameters
        -------
            point1: 2D point with coords [x,y]

            point2: 2D point with coords [x,y]

        Returns
        -------
            flaot: square root of distance of point1 and point2
        """

        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.hypot(dx,dy)

    def _normalize_angle(self, angle: float) -> float:
        """
        Helper method used for ensuring engle is in range (-pi,pi]

        Returns
        -------
            float: normalized angle in range (-pi,pi]
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def _rotate_towards_point(self,target_delta_yaw: float, angular_speed: float = ANGULAR_TO_THE_POINT) -> bool:
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

            dyaw = self._normalize_angle(odom[2] - start_yaw)

            angle_error = self._normalize_angle(target_delta_yaw - dyaw)

            if abs(angle_error) < 0.05:  # ~3 degree tolerance
                break

            # slow down near the end of rotation
            angle_error = self._normalize_angle(target_delta_yaw - dyaw)
            # print(f"start_yaw={start_yaw:.2f}, current_yaw={odom[2]:.2f}, dyaw={dyaw:.2f}, target_dyaw={target_delta_yaw:.2f}, angle_error={angle_error:.2f}")

            angular = KP_ANG * angle_error   # proportional gain
            angular = max(min(angular, ANGULAR_TO_THE_POINT_CLAMP), -ANGULAR_TO_THE_POINT_CLAMP)  # clamp
            if -MINIMAL_ANGULAR < angular < MINIMAL_ANGULAR:
                angular = MINIMAL_ANGULAR if angular > 0 else -MINIMAL_ANGULAR

            self.robot.cmd_velocity(0, angular)
            angular_speed=angular

        self.robot.cmd_velocity(0, 0)
        return True

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

            # distance to goal
            distance = self._distance_from(current, [dest_x, dest_y])

            # desired heading
            desired_yaw = math.atan2(dest_y - y, dest_x - x)

            # heading error
            angle_error = self._normalize_angle(desired_yaw - yaw)

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

        delta_yaw = self._normalize_angle(target_angle - current_yaw)

        # ROtate towards point
        angular_speed = ANGULAR_TO_THE_POINT if delta_yaw > 0 else -ANGULAR_TO_THE_POINT            
        if not self._rotate_towards_point(delta_yaw, angular_speed=angular_speed):
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