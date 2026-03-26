from robolab_turtlebot import Turtlebot
import numpy as np
import cv2
import math

EXIT_ANGULAR_VELOCITY = 0.3
import numpy as np
import cv2

EXIT_ANGULAR_VELOCITY = 0.3

DISTANCE_TOL = 0.085
SPEED_TO_THE_POINT = 0.3
ANGULAR_TO_THE_POINT = 0.7


class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 
        self.robot.reset_odometry()

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        self.stop = False
        #self.exit_garage()
        #self.approach_pylon()
        #self.drive_around_pylon()
        #self.return_to_garage()
        cv2.waitKey(200)  # let it settle
        origin = self.robot.get_odometry()
        self._go_to_point_using_odometry(0, 0)
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
            print("Could not exit the garage!")

    def find_exit(self) -> bool:
        """
        Rotate the robot until a free direction is detected.

        The function analyzes the depth point cloud and estimates the
        distance to obstacles in front of the robot. If sufficient free
        space (>= DISTANCE_DETECTION m) is detected, the robot stops rotating and the
        function returns True.

        Returns
        -------
        bool
            True if a suitable exit direction was found.
            False if the ROS node shuts down before detection.
        """

        DISTANCE_DETECTION = 0.6
        exit_found = False        

        print('Waiting for point cloud ...')
        self.robot.wait_for_point_cloud()
        print('First point cloud recieved ...')

        # WINDOW = 'obstacles' # Name of the OpenCV display window used to visualize the processed point-cloud image
        # cv2.namedWindow(WINDOW)

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

            # mask out floor points
            mask = pc[:, :, 1] < 0.2

            # mask point too far
            mask = np.logical_and(mask, pc[:, :, 2] < 3.0)

            #if np.count_nonzero(mask) <= 0:
            #    print('All point are too far ...')
            #    continue

            # # empty image
            # image = np.zeros(mask.shape)

            # # assign depth i.e. distance to image
            # image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)
            # im_color = cv2.applyColorMap(255 - image.astype(np.uint8),
            #                             cv2.COLORMAP_JET)

            # # show image
            # cv2.imshow(WINDOW, im_color)
            # cv2.waitKey(1)

            # check obstacle
            mask = np.logical_and(mask, pc[:, :, 1] > -0.2)
            data = np.sort(pc[:, :, 2][mask])

            if data.size > 50:
                dist = np.percentile(data, 10)
                if dist >= DISTANCE_DETECTION:
                    exit_found = True

            # exit found
            if exit_found:
                print("Exit found!")
                self.robot.cmd_velocity(0, 0)
                return True

            # rotate to find exit
            else:
                self.robot.cmd_velocity(linear=0, angular=EXIT_ANGULAR_VELOCITY)

        return exit_found


    def drive_out_of_garage(self) -> None:
        """
        The robot drives straight out a short distance in front of the garage.
        Fixed distance.
        
        Returns
        -------
            None
        """
        duration = 3.0      # seconds
        speed = 0.2         # m/s

        start_time = cv2.getTickCount() / cv2.getTickFrequency()

        print("Driving out of garage")
        while not self.robot.is_shutting_down() or not self.stop:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            if current_time - start_time >= duration:
                break

            if not self.stop:
                self.robot.cmd_velocity(linear=speed, angular=0)
            else:
                self.robot.cmd_velocity(0, 0)

        self.robot.cmd_velocity(0, 0)
        print("Out of garage!")

    def approach_pylon(self) -> None:
        """
        Finds the ball and drives to it to a certain distance.
        """
        self.find_pylon()

    def find_pylon(self) -> None:
        """
        Turns the robot in place so that the ball is directly in front of it.
        """
        pass

    def drive_around_pylon(self) -> bool:
        """
        The robot performs a predefined circle maneuver.
        """
        if self.stop:
            self.robot.cmd_velocity(0, 0)
            return

        def drive_for(duration, linear, angular):
            start = cv2.getTickCount() / cv2.getTickFrequency()
            while not self.robot.is_shutting_down() or self.stop:
                now = cv2.getTickCount() / cv2.getTickFrequency()

                if now - start >= duration:
                    break
                
                self.robot.cmd_velocity(linear=linear, angular=angular)
                cv2.waitKey(1)
            self.robot.cmd_velocity(0, 0)

        drive_for(0.9, 0.0, 0.3)
        drive_for(3.0, 0.2, 0.0)
        drive_for(4.0, 0.2, 0.8)
        drive_for(0.9, 0.0, 0.3)
        drive_for(3.0, 0.2, 0.0)

    def return_to_garage(self) -> None:
        """
        The robot finds the garage door, drives in front of it, and then parks inside the garage.
        """
        self.find_garage_entrance()
        self.approach_garage()
        self.park_into_garage()

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
        pass

    def park_into_garage(self) -> None:
        """
        The robot is already in front of the garage door and now simply drives inside.
        """
        pass

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
        Helper method. Local wrapper around self.robot.cmd_velocity(). Checks the self.stop flag.
        Method utilizes odometry to reliably turn the desired amount. 

        Parameters
        -------
            target_delta_yaw: desired change in orintation ( yaw ), units = [rad], absolute value

            angular_speed: angular velocity, units = [rad/s]
                           positive -> counterclokwise
                           negative -> clockwise
        Returns
        -------
            None
        """
        print(f"Rotaing towards point {target_delta_yaw:.2f} with angular speed {angular_speed:.2f}")

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
            print(f"start_yaw={start_yaw:.2f}, current_yaw={odom[2]:.2f}, dyaw={dyaw:.2f}, target_dyaw={target_delta_yaw:.2f}, angle_error={angle_error:.2f}")

            angular = 2.0 * angle_error   # proportional gain
            angular = max(min(angular, 0.5), -0.5)  # clamp

            self.robot.cmd_velocity(0, angular)
            angular_speed=angular

        self.robot.cmd_velocity(0, 0)
        return True

    def _drive_to_the_point(self, dest_x: float, dest_y: float, speed: float = SPEED_TO_THE_POINT) -> bool:
        print(f"Driving straight to point: ({dest_x:.2f}, {dest_y:.2f})")

        Kp_ang = 2.0   # proportional gain for heading correction
        max_ang = 0.5  # angular velocity clamp

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

            print(f"Position: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}), "
                f"distance={distance:.2f}, angle_error={angle_error:.2f}")

            # stop condition
            if distance < DISTANCE_TOL:
                self.robot.cmd_velocity(0, 0)
                return True

            # proportional angular correction
            angular = Kp_ang * angle_error
            angular = max(min(angular, max_ang), -max_ang)  # Clamp

            # optional: slow down when badly misaligned
            linear = speed
            if abs(angle_error) > 0.5:  # ~30 degrees
                linear = 0.0  # rotate in place if very off

            self.robot.cmd_velocity(linear, angular)

        return False

    def _go_to_point_using_odometry(self, dest_x: float, dest_y: float) -> bool:
        print(f"Driving to point: ({dest_x:.2f}, {dest_y:.2f})")

        while not self.robot.is_shutting_down():
            if self.stop:   
                self.robot.cmd_velocity(0, 0)
                return False
            
            current_odom = self.robot.get_odometry()
            if current_odom is None:
                continue # potential problem, robot might stop for a moment

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