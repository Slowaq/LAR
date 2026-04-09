"""Utility functions for vector operations, coordinate transformations,
and mathematical computations."""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Optional


def get_average_of_nearby_pixels(
    pc: np.ndarray, y: int, x: int, window_size: int = 5
) -> Optional[np.ndarray]:
    """
    Compute the average 3D point in a window around (y, x), filtering out
    points further than 0.1 from the center pixel (or middle column).

    Args:
        pc (np.ndarray): HxWx3 point cloud array.
        y (int): Pixel row coordinate.
        x (int): Pixel column coordinate.
        window_size (int, optional): Odd number defining the window size.
            Defaults to 5.

    Returns:
        Optional[np.ndarray]: An array representing the [x, y, z] average, or
            None if no valid points are found.
    """
    half_w = window_size // 2
    h, w, _ = pc.shape

    valid_points = []
    center_p = None
    middle_col_points = []

    # --- First Pass: Collect valid points and identify reference points ---
    for dy in range(-half_w, half_w + 1):
        for dx in range(-half_w, half_w + 1):
            ny = y + dy
            nx = x + dx

            # Bounds check
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            p = pc[ny, nx]

            # Validity checks (filters out NaNs, Infs, and [0,0,0])
            if not np.all(np.isfinite(p)) or np.allclose(p, 0):
                continue

            valid_points.append(p)

            # Store references for the distance check
            if dx == 0:
                middle_col_points.append(p)
                if dy == 0:
                    center_p = p

    if not valid_points:
        return None

    # --- Determine the reference points for distance filtering ---
    if center_p is not None:
        ref_points = [center_p]
    elif middle_col_points:
        ref_points = middle_col_points
    else:
        # If both the center and the entire middle column are invalid/empty,
        # we have no reference to compare against.
        return None

    # --- Second Pass: Filter based on 0.1 distance threshold ---
    filtered_points = []
    for p in valid_points:
        # Check Euclidean distance between the point and the reference point(s)
        # If it's <= 0.1 to ANY valid reference point, we keep it.
        distances = [np.linalg.norm(p - ref) for ref in ref_points]

        if min(distances) <= 0.1:
            filtered_points.append(p)

    if len(filtered_points) == 0:
        return None

    return np.mean(filtered_points, axis=0)


def get_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Args:
        point1 (Tuple[float, float]): The first 2D point [x, y].
        point2 (Tuple[float, float]): The second 2D point [x, y].

    Returns:
        float: The square root of the distance between point1 and point2.
    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.hypot(dx, dy)


def normalize_angle(angle: float) -> float:
    """
    Normalizes an angle to the range [-pi, pi).

    Args:
        angle (float): The input angle in radians.

    Returns:
        float: The normalized angle in the range [-pi, pi).
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotate_vector(x: float, y: float, phi: float) -> Tuple[float, float]:
    """
    Rotate a 2D vector (x, y) counterclockwise by an angle phi.

    Args:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        phi (float): The angle of rotation in radians.

    Returns:
        Tuple[float, float]: The rotated coordinates (x_new, y_new).
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    x_new = x * cos_phi - y * sin_phi
    y_new = x * sin_phi + y * cos_phi

    return x_new, y_new


def average_vector(
    vec1: Tuple[float, float], vec2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate the mathematical average of two 2D vectors.

    Args:
        vec1 (Tuple[float, float]): The first vector.
        vec2 (Tuple[float, float]): The second vector.

    Returns:
        Tuple[float, float]: The averaged 2D vector.
    """
    return ((vec1[0] + vec2[0]) / 2, (vec1[1] + vec2[1]) / 2)


def multiply_vector(
    vec: Tuple[float, float], multiplier: float
) -> Tuple[float, float]:
    """
    Multiply a 2D vector by a scalar value.

    Args:
        vec (Tuple[float, float]): The input vector.
        multiplier (float): The scalar value to multiply the vector by.

    Returns:
        Tuple[float, float]: The scaled vector.
    """
    return (multiplier * vec[0], multiplier * vec[1])


def extend_vector(
    vec: Tuple[float, float], extension: float
) -> Tuple[float, float]:
    """
    Extend the length of a vector by a specified absolute amount while
    maintaining its direction.

    Args:
        vec (Tuple[float, float]): The input vector.
        extension (float): The absolute length to add to the vector.

    Returns:
        Tuple[float, float]: The extended vector.
    """
    current_length = get_distance(vec, (0.0, 0.0))
    if current_length == 0:
        return (0.0, 0.0)  # Added safety check for zero division
    desired_length = current_length + extension
    multiplier = desired_length / current_length
    return multiply_vector(vec, multiplier)


def local_coords_to_global_coords(
    pc_x: float, pc_y: float, odometry: np.ndarray
) -> Tuple[float, float]:
    """
    Convert local robot-centric coordinates to global coordinate frame.

    Assumptions:
        - pc_x is positive to the right of the robot.
        - pc_y is positive in front of the robot.
        - global_x is positive forward.
        - global_y is positive to the left.

    Args:
        pc_x (float): The local x-coordinate.
        pc_y (float): The local y-coordinate.
        odometry (np.ndarray): Array containing
            [robot_global_x, robot_global_y, robot_yaw].

    Returns:
        Tuple[float, float]: The transformed (global_x, global_y) coordinates.
    """
    robot_yaw = odometry[2]
    robot_global_x, robot_global_y = odometry[0], odometry[1]

    x_rotated, y_rotated = rotate_vector(pc_x, pc_y, robot_yaw)

    # Permutation of bases plus translation to robot global position
    point_global_x = robot_global_x + y_rotated
    point_global_y = robot_global_y - x_rotated     # sign changes

    return point_global_x, point_global_y


def global_coords_to_local_coords(
    global_x: float, global_y: float, odometry: np.ndarray
) -> Tuple[float, float]:
    """
    Convert global coordinates back to the local robot-centric frame.

    Args:
        global_x (float): The global x-coordinate.
        global_y (float): The global y-coordinate.
        odometry (np.ndarray): Array containing
            [robot_global_x, robot_global_y, robot_yaw].

    Returns:
        Tuple[float, float]: The transformed (local_x, local_y) coordinates.
    """
    robot_global_x, robot_global_y, robot_yaw = (
        odometry[0], odometry[1], odometry[2]
    )

    # 1. Translate back to robot-centric origin
    dx = global_x - robot_global_x
    dy = global_y - robot_global_y

    # 2. Revert the axis mapping/sign changes
    y_rotated = dx
    x_rotated = -dy

    # 3. Rotate back by the negative yaw
    pc_x, pc_y = rotate_vector(x_rotated, y_rotated, -robot_yaw)

    return pc_x, pc_y


def clamp_speed(
    speed: float, max_speed: float, min_speed: float = 0.0
) -> float:
    """
    Clamp the speed to the range [-max_speed, max_speed].

    Args:
        speed (float): The input speed value.
        max_speed (float): The maximum allowed speed.
        min_speed (float): The minimum allowed speed.

    Returns:
        float: The clamped speed value.
    """
    if speed > max_speed:
        return max_speed
    elif speed < -max_speed:
        return -max_speed
    elif 0 < speed < min_speed:
        return min_speed
    elif -min_speed < speed < 0:
        return -min_speed
    else:
        return speed


def line_intersects_circle(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    circle_center: Tuple[float, float],
    radius: float
) -> bool:
    """Detect if a line segment intersects a circle.

    Use projection to find the closest point on the line segment to the
    circle center. If the distance is within the radius, an intersection
    exists.

    Args:
        point_1 (Tuple[float, float]): First endpoint of the line segment.
        point_2 (Tuple[float, float]): Second endpoint of the line segment.
        circle_center (Tuple[float, float]): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        bool: True if the line segment intersects the circle, False
            otherwise.
    """
    x1, y1 = point_1
    x2, y2 = point_2
    xc, yc = circle_center

    # Vector AB (from point_1 to point_2)
    dx = x2 - x1
    dy = y2 - y1

    # Squared length of the line segment AB
    len_sq = dx**2 + dy**2

    # Edge case: point_1 and point_2 are the exact same point
    if len_sq == 0:
        dist = math.hypot(xc - x1, yc - y1)
        return dist <= radius

    # Vector AC (from point_1 to circle_center)
    ac_x = xc - x1
    ac_y = yc - y1

    # Calculate the projection scalar t
    # t = (AC dot AB) / (AB dot AB)
    t = (ac_x * dx + ac_y * dy) / len_sq

    # Check if the center projects outside the line segment
    if t < 0.0 or t > 1.0:
        return False

    # Find the projected point P on the line segment
    px = x1 + t * dx
    py = y1 + t * dy

    # Calculate the distance from the circle center to the projected point P
    dist = math.hypot(xc - px, yc - py)

    # Intersection occurs if the distance is less than or equal to the radius
    return dist <= radius
