# import math

# def calculate_angle(a, b, c):
#     """
#     Calculates the angle (in degrees) between three points.
#     Parameters:
#     - a, b, c: landmarks with x, y coordinates
#     Returns:
#     - angle in degrees
#     """
#     # Get the coordinates
#     x1, y1 = a.x, a.y
#     x2, y2 = b.x, b.y
#     x3, y3 = c.x, c.y

#     # Calculate the vectors
#     v1 = [x1 - x2, y1 - y2]
#     v2 = [x3 - x2, y3 - y2]

#     # Calculate the angle between the vectors
#     angle_rad = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
#     angle_deg = math.degrees(angle_rad)

#     # Ensure the angle is positive
#     if angle_deg < 0:
#         angle_deg += 360

#     return angle_deg

import math

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points.
    The angle is calculated at point b, with the line segments formed by (a-b) and (b-c).
    """
    # Get the coordinates
    a_x, a_y = a.x, a.y
    b_x, b_y = b.x, b.y
    c_x, c_y = c.x, c.y

    # Create vectors
    ab = [a_x - b_x, a_y - b_y]  # Vector from b to a
    bc = [c_x - b_x, c_y - b_y]  # Vector from b to c

    # Calculate dot product and magnitudes
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]  # Dot product of vectors
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Avoid division by zero
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0

    # Calculate angle in radians and convert to degrees
    angle_rad = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    angle_deg = math.degrees(angle_rad)

    return angle_deg

