from .algorithm import Algorithm
from dataclasses import dataclass

STATE_RELEASED = 0
STATE_PRESSED = 1

BUMPER_LEFT = 0
BUMPER_CENTER = 1
BUMPER_RIGHT = 2

BUTTON_B0 = 0
BUTTON_B1 = 1
BUTTON_B2 = 2

# TODO - use proper class msg class instead of a mock
@dataclass
class BumperMsgMock:
    """
    Mock message representing a bumper event.

    Attributes:
        bumper (int): Identifier of the bumper that triggered the event
            (e.g., BUMPER_LEFT, BUMPER_CENTER, BUMPER_RIGHT).
        state (int): State of the bumper (STATE_PRESSED or STATE_RELEASED).
    """
    bumper: int
    state: int

@dataclass
class ButtonMsgMock:
    """
    Mock message representing a button event.

    Attributes:
        button (int): Identifier of the button that triggered the event
            (e.g., BUTTON_B0, BUTTON_B1, BUTTON_B2).
        state (int): State of the button (STATE_PRESSED or STATE_RELEASED).
    """
    button: int
    state: int

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
    """

    def _bumper_cb(msg: BumperMsgMock) -> None:
        """
        Handle bumper events.

        Stops the algorithm when any bumper is pressed.

        Args:
            msg (BumperMsgMock): Incoming bumper event message.
        """
        if msg.state == STATE_PRESSED:
            print("Bumper pressed - stopping execution")
            algorithm.stop = True

    def _button_cb(msg: ButtonMsgMock) -> None:
        """
        Handle button events.

        - BUTTON_B0 press triggers algorithm execution.
        - BUTTON_B1 press stops the algorithm.

        Args:
            msg (ButtonMsgMock): Incoming button event message.
        """
        if msg.state == STATE_PRESSED:
            if msg.button == BUTTON_B0:
                print("Button B0 pressed - starting execution")
                algorithm.run()
            elif msg.button == BUTTON_B1:
                print("Button B1 pressed - stopping execution")
                algorithm.stop = True

    algorithm.robot.register_bumper_event_cb(_bumper_cb)
    algorithm.robot.register_button_event_cb(_button_cb)