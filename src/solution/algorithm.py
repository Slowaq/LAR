from robolab_turtlebot import Turtlebot

class Algorithm:
    def __init__(self):
        self.robot = Turtlebot(rgb=True, depth=True)
        self.stop : bool = False    # When a bumper hits something or a button is pressed, the robot stops. 

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