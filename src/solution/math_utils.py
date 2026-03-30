import math

def distance(point1: list[int, int], point2: list[int, int]) -> float:
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
    Normalizes angle to (-pi, pi].

    Returns
    -------
        float: normalized angle in range (-pi,pi]
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi