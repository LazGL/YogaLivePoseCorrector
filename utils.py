import math

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points.
    Parameters:
    - a, b, c: landmarks with x, y coordinates
    Returns:
    - angle in degrees
    """
    # Get the coordinates
    x1, y1 = a.x, a.y
    x2, y2 = b.x, b.y
    x3, y3 = c.x, c.y

    # Calculate the vectors
    v1 = [x1 - x2, y1 - y2]
    v2 = [x3 - x2, y3 - y2]

    # Calculate the angle between the vectors
    angle_rad = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    angle_deg = math.degrees(angle_rad)

    # Ensure the angle is positive
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

