import math


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)