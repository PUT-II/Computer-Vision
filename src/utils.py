from typing import Tuple

import cv2 as cv
import numpy as np


def get_longest_line_points(contour) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ Finds longest line in contour

    :param contour:
    :return: Tuple with start and end point of line
    """
    result: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (1, 2))
    result_length: float = 0.0

    for j in range(-1, len(contour) - 1):
        length: float = calculate_distance(contour[j][0], contour[j + 1][0])
        if length > result_length:
            result = contour[j][0], contour[j + 1][0]
            result_length = length

    return tuple(result[0]), tuple(result[1])


def calculate_distance(p_1: Tuple[int, int], p_2: Tuple[int, int]) -> float:
    return ((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2) ** 0.5


def find_contour(image: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    longest_contour = max(contours, key=lambda contour: len(contour))
    return longest_contour
