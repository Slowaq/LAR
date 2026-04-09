"""Callback functions for handling bumper and button events."""

from .algorithm import Algorithm
from kobuki_msgs.msg import ButtonEvent, BumperEvent
import threading

STATE_RELEASED = 0
STATE_PRESSED = 1

BUMPER_LEFT = 0
BUMPER_CENTER = 1
BUMPER_RIGHT = 2

BUTTON_B0 = 0
BUTTON_B1 = 1
BUTTON_B2 = 2


def register_callbacks(algorithm: Algorithm) -> None:
    """Register bumper and button event callbacks on the robot.

    Callbacks control the algorithm execution state with threaded execution:
    - Any bumper press sets the stop flag to True.
    - Button B0 triggers non-blocking execution in a background thread.
    - Button B1 sets the stop flag to True.
    - Duplicate B0 presses are ignored while an execution thread is active.

    Args:
        algorithm (Algorithm): The algorithm instance to control.
    """

    def _bumper_cb(msg: BumperEvent) -> None:
        """Handle bumper events to halt robot movement.

        Args:
            msg (BumperEvent): Incoming bumper event message.
        """
        if msg.state == STATE_PRESSED and not algorithm.stop:
            print("Bumper pressed - stopping execution")
            algorithm.stop = True

    def _run_with_guard() -> None:
        """Manage the algorithm lifecycle in a dedicated thread.

        Ensures the 'is_running' flag is cleared and the 'stop' flag is
        set upon completion or failure, allowing for subsequent restarts.
        """
        try:
            algorithm.run()
        except Exception as e:
            print(f"ALGORITHM CRASHED: {e}")
        finally:
            algorithm.is_running = False
            algorithm.stop = True

    def _button_cb(msg: ButtonEvent) -> None:
        """Handle button events using a non-blocking threaded approach.

        If the algorithm is already running, B0 events are discarded to
        prevent ROS callback queueing from triggering multiple sequential
        runs. B1 remains active to allow user-interruption.

        Args:
            msg (ButtonEvent): Incoming button event message.
        """

        if msg.state == STATE_PRESSED:

            # 1. Guard: If already running, handle stop signals or
            # ignore start signals
            if algorithm.is_running:
                if msg.button == BUTTON_B1:
                    print("Button B1 pressed - stopping execution")
                    algorithm.stop = True
                return  # Exit to discard any queued messages
                # (like multiple B0 presses)

            # 2. Trigger execution if idle
            if msg.button == BUTTON_B0:
                print("Button B0 pressed - starting execution in background")
                algorithm.stop = False
                algorithm.is_running = True

                # Offload long-running .run() to a thread to keep the
                # callback responsive
                thread = threading.Thread(target=_run_with_guard)
                thread.daemon = True
                thread.start()

    algorithm.robot.register_bumper_event_cb(_bumper_cb)
    algorithm.robot.register_button_event_cb(_button_cb)
