import cv2, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from time import sleep

from src.solution import *

from sys import argv


def main() -> None:
    algorithm = Algorithm()
    register_callbacks(algorithm)
    while (not algorithm.robot.is_shutting_down()):
        sleep(0.1)
    return

if __name__ == "__main__":
    main()