from src.solution import *
import cv2, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from time import sleep
from robolab_turtlebot import Turtlebot

from sys import argv
import time

# Globals for mouse callback
bgr_img = None
hsv_img = None
display_img = None

def mouse_callback(event, x, y, flags, param):
    global display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_img[y, x]
        bgr_value = bgr_img[y, x]
        text = f"HSV: {hsv_value} | BGR: {bgr_value}"
        print(f"Clicked at ({x}, {y}) -> {text}")

        # Draw a small circle where clicked
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display_img,
                    text,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

def process_frame(bgr_img, pc):
    # --- DEPTH VISUALIZATION ---
    image = np.zeros(pc.shape[:2])

    mask_depth = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)
    image[mask_depth] = np.int8(pc[:, :, 2][mask_depth] / 3.0 * 255)

    depth_vis = cv2.applyColorMap(
        255 - image.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # --- PURPLE DETECTION ---
    centers, annotated_bgr, frame_bw = find_purple_quads(bgr_img)
    # print(centers)

    for (x, y) in centers:
        if 0 <= y < pc.shape[0] and 0 <= x < pc.shape[1]:
            distance = pc[y, x, 2]
            if not np.isnan(distance):
                cv2.putText(annotated_bgr,
                            f"{distance:.2f} m",
                            (x - 40, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1)

    # Convert BW mask to 3 channels for visualization
    mask_vis = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)

    # Combine annotated RGB, depth, and mask
    combined = np.hstack((annotated_bgr, depth_vis, mask_vis))
    return combined

def debug_from_files(rgb_path, pc_path):
    global bgr_img, hsv_img, display_img

    bgr_img = cv2.imread(rgb_path)
    if bgr_img is None:
        print(f"Failed to read {rgb_path}")
        return

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    display_img = bgr_img.copy()

    # Load point cloud
    pc = np.load(pc_path)

    cv2.namedWindow("RGB + Depth (DEBUG)")
    cv2.setMouseCallback("RGB + Depth (DEBUG)", mouse_callback)

    combined = process_frame(bgr_img, pc)
    display_img = combined.copy()

    while True:
        cv2.imshow("RGB + Depth (DEBUG)", display_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

def main() -> None:
    global bgr_img, hsv_img, display_img

    if len(argv) == 3:
        debug_from_files(argv[1], argv[2])
        return

    # NORMAL ROBOT MODE
    robot = Turtlebot(True, True, True)
    robot.wait_for_rgb_image()
    robot.wait_for_point_cloud()

    cv2.namedWindow("RGB + Depth")
    cv2.setMouseCallback("RGB + Depth", mouse_callback)

    while not robot.is_shutting_down():
        time.sleep(0.1)

        # Get BGR image directly
        bgr_img = robot.get_rgb_image()  # already RGB from robot
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        pc = robot.get_point_cloud()

        combined = process_frame(bgr_img, pc)
        display_img = combined.copy()

        cv2.imshow("RGB + Depth", display_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()