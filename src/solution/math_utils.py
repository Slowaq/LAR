import math
import numpy as np

def get_average_of_nearby_pixels(pc, y, x, window_size=5):
    """
    Compute the average 3D point in a window around (y, x).

    Args:
        pc: HxWx3 point cloud (numpy array)
        y, x: pixel coordinates, row, column (int)
        window_size: odd number (default 5 for 5x5 window)

    Returns:
        np.array([x, y, z]) average or None if no valid points found
    """
    half_w = window_size // 2
    h, w, _ = pc.shape

    points = []

    for dy in range(-half_w, half_w + 1):
        for dx in range(-half_w, half_w + 1):
            ny = y + dy
            nx = x + dx

            # Bounds check
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            p = pc[ny, nx]

            # Validity checks
            if not np.all(np.isfinite(p)):
                continue

            if np.allclose(p, 0):
                continue

            points.append(p)

    if len(points) == 0:
        return None

    return np.mean(points, axis=0)

def get_distance(point1: list, point2: list) -> float:
    """
    Calculate distance from two 2D points.

    Parameters
    -------
        point1: 2D point with coords [x,y]

        point2: 2D point with coords [x,y]

    Returns
    -------
        flaot: square root of distance of point1 and point2
    """

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.hypot(dx,dy)

def normalize_angle(angle: float) -> float:
    """
    Normalizes angle to [-pi, pi).

    Returns
    -------
        float: normalized angle in range [-pi,pi)
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotate_vector(x, y, phi):
    """
    Rotate a 2D vector (x, y) counterclockwise by angle phi (radians).

    Returns:
        (x_new, y_new): tuple of rotated coordinates
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    x_new = x * cos_phi - y * sin_phi
    y_new = x * sin_phi + y * cos_phi
    
    return x_new, y_new

def average_vector(vec1: tuple, vec2: tuple) -> tuple:
    return ((vec1[0] + vec2[0])/2, (vec1[1] + vec2[1])/2)

def substract_vectors(vec1: tuple, vec2: tuple) -> tuple:
    return (vec1[0] - vec2[0], vec1[1] - vec2[1])

def multiply_vector(vec: tuple, multiplier: float) -> tuple:
    return (multiplier * vec[0], multiplier * vec[1])

def normalize_vector(vec: tuple) -> tuple:
    lenght = get_distance(vec, (0,0))
    return multiply_vector(vec, 1/lenght)

def dot_product(vec1: tuple, vec2: tuple) -> float:
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]