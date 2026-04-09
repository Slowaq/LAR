"""Entry point script for running the TurtleBot algorithm."""

from robolab_turtlebot import sleep
from src.solution import Algorithm, register_callbacks


def main() -> None:
    """Initialize the algorithm, register callbacks, and run the main loop."""
    algorithm = Algorithm()
    register_callbacks(algorithm)
    while (not algorithm.robot.is_shutting_down()):
        sleep(0.1)
    return


if __name__ == "__main__":
    main()
