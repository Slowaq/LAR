from robolab_turtlebot import Turtlebot
import numpy as np
import cv2

class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 
        self.exit_angular_vel = 0.3  # How fast the robot rotates during finding exit

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        self.wait_for_start_button()
        self.exit_garage()
        self.approach_pylon()
        self.drive_around_pylon()
        self.return_to_garage()

    def wait_for_start_button(self) -> None:
        """
        The program waits until the start button is pressed.
        """

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
                self.robot.cmd_velocity(0, 0)
                print("Exit found!")
                return True

            # rotate to find exit
            else:
                self.robot.cmd_velocity(linear=0, angular=self.exit_angular_vel)

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
        while not self.robot.is_shutting_down():
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

    def drive_around_pylon(self) -> None:
        """
        The robot performs a predefined circle maneuver.
        """
        pass

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

    def _turn_on_the_spot(self) -> None:
        """
        Helper method. Local wrapper around self.robot.cmd_velocity(). Checks the self.stop flag.
        """
        pass
