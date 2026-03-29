import numpy as np
import cv2

CIRCULARITY_THRESHOLD = 0.7

def find_pylon(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 70, 70])
    upper_green = np.array([90, 255, 255])

    # create a mask for green color and discard noise
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # sort contours by area from largest to smallest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            # check circularity(roundness) of the contour
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < CIRCULARITY_THRESHOLD:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            x, y = int(x), int(y)

            width = frame.shape[1]
            center_x = width // 2 + 50
            error = x - center_x

            # kreslenie
            cv2.circle(frame_bgr, (x, y), int(radius), (0,255,0), 2)
            cv2.circle(frame_bgr, (x, y), 3, (0,0,255), -1)
            cv2.line(frame_bgr, (center_x, 0),
                     (center_x, frame.shape[0]), (255,0,0), 2)

            print(f"Found pylon: error={error:.2f}, x={x:.2f}, y={y:.2f}")
            return error, x, y, frame_bgr

    print("Couldnt find pylon")
    return None, None, None, frame_bgr