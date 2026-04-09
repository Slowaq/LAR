"""Image segmentation utilities for detecting pylons and purple quads."""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Optional, Tuple

CIRCULARITY_THRESHOLD = 0.52
ASPECT_RATIO_LOWER_THRESHOLD = 0.65
ASPECT_RATIO_UPPER_THRESHOLD = 1.55
ASPECT_RATIO_STRICT_LOWER_THRESHOLD = 0.8
ASPECT_RATIO_STRICT_UPPER_THRESHOLD = 1.25


def find_pylon(
    frame: np.ndarray
) -> Tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Detect the green pylon in an RGB frame.

    Args:
        frame (np.ndarray): BGR image from the robot's camera.

    Returns:
        tuple:
            Optional[Tuple[int, int]]: pixel coordinates (column, row) of the
                detected pylon center, or None if no pylon was found.
            np.ndarray: visualization image with overlays drawn.
            np.ndarray: binary mask used for the green detection.
    """
    frame_bgr = frame.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 60, 57])
    upper_green = np.array([90, 245, 255])

    # create a mask for green color and discard noise
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Apply the vertical offset: set top 80 rows to 0 (black)
    mask[0:80, :] = 0

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    target_coords = None

    if contours:
        # sort contours by area from largest to smallest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        passed_area_check = True
        passed_circularity_check = True
        passed_aspect_ratio_check = True

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 10000:
                passed_area_check = False

            aspect_ratio_lower = (
                ASPECT_RATIO_STRICT_LOWER_THRESHOLD
                if area > 2000
                else ASPECT_RATIO_LOWER_THRESHOLD
            )
            aspect_ratio_upper = (
                ASPECT_RATIO_STRICT_UPPER_THRESHOLD
                if area > 2000
                else ASPECT_RATIO_UPPER_THRESHOLD
            )

            # check circularity(roundness) of the contour
            perimeter = cv2.arcLength(cnt, True)
            circularity = (
                4 * np.pi * area / (perimeter ** 2)
                if perimeter > 0
                else 0
            )

            if circularity < CIRCULARITY_THRESHOLD:
                passed_circularity_check = False

            # check the aspect ratio of the rectangle around the ball
            bx, by, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h

            if (
                aspect_ratio < aspect_ratio_lower
                or aspect_ratio > aspect_ratio_upper
            ):
                passed_aspect_ratio_check = False

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            x, y = int(x), int(y)

            # Save the coordinates of the largest contour (the first one
            # processed)
            if (
                target_coords is None
                and passed_area_check
                and passed_circularity_check
                and passed_aspect_ratio_check
            ):
                cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)
                target_coords = (x, y)

            # kreslenie - draw bounding circle and center for EVERY contour
            cv2.circle(frame_bgr, (x, y), int(radius), (0, 255, 0), 2)

            label = (
                f"A:{area} C:{circularity:.2f}, R:{aspect_ratio:.2f}"
            )
            cv2.putText(
                frame_bgr, label, (x + 10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            checks = (
                f"A:{passed_area_check} C: {passed_circularity_check}, "
                f"R: {passed_aspect_ratio_check}"
            )
            cv2.putText(
                frame_bgr, checks, (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

    # Return the target coordinates (if any were found), the drawn frame,
    # and the mask
    return target_coords, frame_bgr, mask


def find_purple_quads(
    frame_bgr: np.ndarray
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Detect purple rectangular quads in a BGR frame.

    Args:
        frame_bgr (np.ndarray): BGR image from the robot's camera.

    Returns:
        tuple:
            List[Tuple[int, int]]: ordered list of quad center coordinates
                as (column, row).
            np.ndarray: annotated BGR image showing detections.
            np.ndarray: binary mask of detected purple regions.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([111, 80, 60])
    upper_purple = np.array([145, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    frame_bw = mask.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue

        # 1. Solidity Check: area of contour / area of convex hull
        # Rectangles should have high solidity (close to 1.0)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < 0.8:  # Filter out "hollow" or complex shapes
            continue

        # 2. Aspect Ratio Check
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # Wh expect tall rectangles
        if aspect_ratio > 0.5:
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

    centers.sort(key=lambda x: abs(x[0] - 320))

    # Draw detected quads
    for poly, (cx, cy) in zip(detected, centers):
        cv2.drawContours(frame_bgr, [poly], -1, (0, 255, 0), 2)
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

    return centers, frame_bgr, frame_bw
