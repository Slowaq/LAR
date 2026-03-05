import numpy as np
import cv2

BALL_HSV_REFERENCE = np.array([74, 129, 110])
HSV_TOLERANCES = np.array([10, 60, 60])

def find_ball_segments(hsv: np.ndarray) -> np.ndarray:
    # Compute absolute differences for H, and thresholds for S and V
    h_diff = np.abs(hsv[:, :, 0].astype(int) - BALL_HSV_REFERENCE[0])
    s_mask = hsv[:, :, 1] > HSV_TOLERANCES[1]
    v_mask = hsv[:, :, 2] > HSV_TOLERANCES[2]

    print(f"pixel HSV (362, 382) is {hsv[382, 362, :]}")

    # Combine all conditions
    mask = (h_diff < HSV_TOLERANCES[0]) & s_mask & v_mask

    # Convert to uint8 black-and-white image
    bw_image = mask.astype(np.uint8) * 255
    print(f"pixel BW (362, 382) is {bw_image[382, 362]}")
    return bw_image