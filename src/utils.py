import math
from typing import List, Tuple


def get_longest_line_points(contour) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ Finds longest line in contour

    :param contour:
    :return: Tuple with start and end point of line
    """
    result: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (1, 2))
    result_length: float = 0.0

    for j in range(-1, len(contour) - 1):
        length: float = calculate_length(contour[j][0], contour[j + 1][0])
        if length > result_length:
            result = contour[j][0], contour[j + 1][0]
            result_length = length

    return tuple(result[0]), tuple(result[1])


def calculate_length(p_1: Tuple[int, int], p_2: Tuple[int, int]) -> float:
    return ((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2) ** 0.5


# Source: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
def calculate_angle(p_1: Tuple[int, int], p_2: Tuple[int, int], p_3: Tuple[int, int]):
    ang = math.degrees(math.atan2(p_3[1] - p_2[1], p_3[0] - p_2[0]) - math.atan2(p_1[1] - p_2[1], p_1[0] - p_2[0]))
    return ang + 360 if ang < 0 else ang


def calculate_angles(contour: List[Tuple[int, int]]) -> List[float]:
    angles: List[float] = []
    for i in range(0, len(contour) - 2):
        p_1 = contour[i]
        p_2 = contour[i + 1]
        p_3 = contour[i + 2]

        angle = calculate_angle(p_1, p_2, p_3)
        angles.append(round(angle, 2))

    return angles
