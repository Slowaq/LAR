from .algorithm import Algorithm
from kobuki_msgs.msg import ButtonEvent, BumperEvent

STATE_RELEASED = 0
STATE_PRESSED = 1

BUMPER_LEFT = 0
BUMPER_CENTER = 1
BUMPER_RIGHT = 2

BUTTON_B0 = 0
BUTTON_B1 = 1
BUTTON_B2 = 2


def register_callbacks(algorithm: Algorithm) -> None:
    """
    Register bumper and button event callbacks on the robot.

    The callbacks control the algorithm execution as follows:
    - Any bumper press will immediately stop the algorithm.
    - Button B0 press will start/run the algorithm.
    - Button B1 press will stop the algorithm.

    Args:
        algorithm (Algorithm): The algorithm instance whose execution
            is controlled via robot input events.

    Returns:
        None
    """

    def _bumper_cb(msg: BumperEvent) -> None:
        """
        Handle bumper events.

        Stops the algorithm when any bumper is pressed.

        Args:
            msg (BumperEvent): Incoming bumper event message.

        Returns:
            None
        """
        if msg.state == STATE_PRESSED and not algorithm.stop:
            print("Bumper pressed - stopping execution")
            algorithm.stop = True

    def _button_cb(msg: ButtonEvent) -> None:
        """
        Handle button events.

        - BUTTON_B0 press triggers algorithm execution.
        - BUTTON_B1 press stops the algorithm.

        Args:
            msg (ButtonEvent): Incoming button event message.

        Returns:
            None
        """
        if msg.state == STATE_PRESSED:
            if msg.button == BUTTON_B0 and algorithm.stop:  # Only when the robot is inactive
                print("Button B0 pressed - starting execution")
                algorithm.run()
            elif msg.button == BUTTON_B1 and not algorithm.stop:
                print("Button B1 pressed - stopping execution")
                algorithm.stop = True

    algorithm.robot.register_bumper_event_cb(_bumper_cb)
    algorithm.robot.register_button_event_cb(_button_cb)
