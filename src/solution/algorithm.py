from robolab_turtlebot import Turtlebot
from .segmentation import *

class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True, pc=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 

    def run(self) -> None:
        """
        This function defines the instruction pipeline for the robot, from starting the program
        to successfully parking in the garage.
        """
        # self.wait_for_start_button()
        # self.exit_garage()
        self.approach_pylon()
        # self.drive_around_pylon()
        # self.return_to_garage()

    def wait_for_start_button(self) -> None:
        """
        The program waits until the start button is pressed.
        """

    def exit_garage(self) -> None:
        """
        The robot orients itself and exits the garage.
        """
        self.find_exit()
        self.drive_out_of_garage()

    def find_exit(self) -> None:
        """
        Turn in place towards the garage exit.
        Preparation for the robot to drive straight out.
        """
        pass

    def drive_out_of_garage(self) -> None:
        """
        The robot drives straight out a short distance in front of the garage.
        Fixed distance.
        """
        pass

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

            combined = np.hstack((frame, depth_vis))
            cv2.imshow("combined", combined)
            cv2.waitKey(1)

        cv2.destroyAllWindows()



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