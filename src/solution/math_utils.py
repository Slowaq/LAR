import math
import numpy as np

import numpy as np

def get_average_of_nearby_pixels(pc, y, x, window_size=5):
    """
    Compute the average 3D point in a window around (y, x), filtering out
    points further than 0.1 from the center pixel (or middle column).

    Args:
        pc: HxWx3 point cloud (numpy array)
        y, x: pixel coordinates, row, column (int)
        window_size: odd number (default 5 for 5x5 window)

    Returns:
        np.array([x, y, z]) average or None if no valid points found
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

def extend_vector(vec: tuple, extension: float) -> tuple:
    current_lenght = get_distance(vec, (0,0))
    desired_lenght = current_lenght + extension
    multiplier = desired_lenght / current_lenght
    return multiply_vector(vec, multiplier)

def local_coords_to_global_coords(pc_x: float, pc_y: float, odometry: np.ndarray) -> tuple:
    """
    pc_x is positive right of the robot
    pc_y is positive in front of the robot
    returns (global_x, global_y), where global x is positive forward and y is positive to the left
    """
    
    robot_yaw = odometry[2]
    robot_global_x, robot_global_y = odometry[0], odometry[1]
    x_rotated, y_rotated =  rotate_vector(pc_x, pc_y, robot_yaw)
    point_global_x = robot_global_x + y_rotated
    point_global_y = robot_global_y - x_rotated     # sign changes
    return point_global_x, point_global_y