import cv2, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from src.solution import *

from sys import argv

BALL_HSV_REFERENCE = [74, 129, 110]

def main() -> None:
    filepath = os.path.join("sample_data", "recording_0001_mic_a_garaz_tma", "0010.mat")
    data = scipy.io.loadmat(filepath)

    rgb = data["image_rgb"]
    hsv = rgb_to_hsv(rgb)

    draw_hsv(hsv)

    segments_image = find_ball_segments(hsv)

    cv2.imshow("Ball Mask", segments_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    main()