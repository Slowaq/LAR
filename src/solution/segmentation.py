import numpy as np
import cv2

def find_pylon(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 70, 70])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 300:
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

def find_purple_quads(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([111, 80, 60])
    upper_purple = np.array([145, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    frame_bw = mask.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        # Compute 4-point bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        detected.append(box)

        # Compute center using moments of the original contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            centers.append((int(rect[0][0]), int(rect[0][1])))  # fallback

    # Draw detected quads
    for poly, (cx, cy) in zip(detected, centers):
        cv2.drawContours(frame_bgr, [poly], -1, (0, 255, 0), 2)
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

    return centers, frame_bgr, frame_bw