import collections
from typing import Tuple, List

import cv2 as cv
import numpy as np

from src.utils import calculate_distance, find_contour


class FragmentMetadata:
    __base_points: Tuple[int, int]
    __cut_edge: List[Tuple[int, int]]
    __sides: Tuple[Tuple[int, int], Tuple[int, int]]
    __image: np.ndarray
    __contour: List[Tuple[int, int]]
    __approx: List[Tuple[int, int]]
    __normalized_cut_points: List[Tuple[int, int]]

    distance_to_base_array: np.array

    def __init__(self, base: Tuple[int, int], sides: tuple, image: np.ndarray, contour: np.ndarray, approx: np.ndarray,
                 cut_point_count: int = 50):
        self.__base_points = base
        self.__sides = sides

        self.__approx = [tuple(point[0]) for point in approx]
        self.__contour = [tuple(point[0]) for point in contour]
        self.__contour = self.__shift_contour()

        self.__image = image

        self.__cut_edge = self.__generate_cut_edge()
        if cut_point_count > 0:
            self.__normalized_cut_points = self.__generate_normalized_cut_points(cut_point_count)
            self.distance_to_base_array = self.__generate_cut_distances(self.__normalized_cut_points)

    def get_side_points(self) -> tuple:
        side_1, side_2 = self.__sides
        side_1_points = (self.__approx[side_1[0]], self.__approx[side_1[1]])
        side_2_points = (self.__approx[side_2[0]], self.__approx[side_2[1]])
        return side_1_points, side_2_points

    def draw_features(self) -> None:
        image = cv.cvtColor(self.__image.copy(), cv.COLOR_GRAY2BGR)

        if len(self.__approx) < 4:
            return

        # Draw base
        base_start = self.__approx[self.__base_points[0]]
        base_end = self.__approx[self.__base_points[1]]
        image = cv.line(image, base_start, base_end, (0, 0, 255), 2)

        # Draw sides
        for i in range(len(self.__sides)):
            side_start = self.__approx[self.__sides[i][0]]
            side_end = self.__approx[self.__sides[i][1]]
            image = cv.line(image, side_start, side_end, (255, 0, 255), 2)

        # Draw cut edge
        for i in range(len(self.__cut_edge) - 1):
            point_1 = self.__cut_edge[i]
            point_2 = self.__cut_edge[i + 1]
            image = cv.line(image, point_1, point_2, (0, 255, 0), 2)

        cut_point_start = self.__cut_edge[0]
        cut_point_end = self.__cut_edge[len(self.__cut_edge) - 1]
        image = cv.line(image, cut_point_start, cut_point_start, (0, 255, 0), 9)
        image = cv.line(image, cut_point_end, cut_point_end, (0, 255, 0), 9)

        cv.imshow("features", image)

        img_copy = cv.cvtColor(self.__image.copy(), cv.COLOR_GRAY2BGR)
        # Draw contour direction
        img_copy = cv.arrowedLine(img_copy, self.__approx[0], self.__approx[1], (0, 255, 255), 3)

        for point in self.__approx:
            img_copy = cv.line(img_copy, point, point, (0, 0, 255), 6)

        cut_ends = []
        for i in self.__sides[0] + self.__sides[1]:
            if i not in self.__base_points:
                cut_ends.append(self.__approx[i])
        for point in cut_ends:
            img_copy = cv.line(img_copy, point, point, (0, 255, 0), 20)

        cv.imshow("features-orig", img_copy)

        img_copy = cv.cvtColor(self.__image.copy(), cv.COLOR_GRAY2BGR)
        for point in self.__normalized_cut_points:
            img_copy = cv.line(img_copy, point, point, (0, 0, 255), 6)

        cv.imshow("features-normalized-points", img_copy)

        img_copy = cv.cvtColor(self.__image.copy(), cv.COLOR_GRAY2BGR)
        for point in self.__contour:
            img_copy = cv.line(img_copy, point, point, (0, 0, 255), 6)

        # Draw contour direction
        img_copy = cv.arrowedLine(img_copy, self.__contour[0], self.__approx[1], (0, 255, 255), 3)
        cv.imshow("features-orig-full", img_copy)

        cv.waitKey()
        cv.destroyAllWindows()

    def __shift_contour(self) -> List[Tuple[int, int]]:
        # Rotate approximated contour
        approx_index = self.__approx.index(self.__approx[self.__base_points[0]])
        deque = collections.deque(self.__approx)
        rot_val = -approx_index
        deque.rotate(rot_val)

        # Correct base and side edges points
        self.__base_points = (
            (self.__base_points[0] + rot_val) % len(self.__approx),
            (self.__base_points[1] + rot_val) % len(self.__approx))
        side_1 = (
            (self.__sides[0][0] + rot_val) % len(self.__approx), (self.__sides[0][1] + rot_val) % len(self.__approx))
        side_2 = (
            (self.__sides[1][0] + rot_val) % len(self.__approx), (self.__sides[1][1] + rot_val) % len(self.__approx))
        self.__sides = (side_1, side_2)
        self.__approx = list(deque)

        # Rotate contour
        cnt_index = self.__contour.index(self.__approx[self.__base_points[0]])
        deque = collections.deque(self.__contour)
        deque.rotate(-cnt_index)
        return list(deque)

    def __generate_normalized_cut_points(self, point_count) -> List[Tuple[int, int]]:
        cut_edge_points = self.__cut_edge
        cut_edge_points = list(sorted(cut_edge_points, key=lambda point: point[0]))

        cut_ends = []
        for i in self.__sides[0] + self.__sides[1]:
            if i not in self.__base_points:
                cut_ends.append(self.__approx[i])

        x_min: int = min(cut_ends, key=lambda point: point[0])[0]
        x_max: int = max(cut_ends, key=lambda point: point[0])[0]
        step = (x_max - x_min) / point_count

        x = x_min - step
        points: List[Tuple[int, int]] = []
        previous_point_index = 0
        while x <= x_max:
            x += step
            first_point = next((point for point in cut_edge_points[previous_point_index:] if point[0] >= x), None)

            if first_point is None:
                break

            previous_point_index = cut_edge_points.index(first_point, previous_point_index)
            second_point_index = previous_point_index + 1

            if second_point_index == len(cut_edge_points):
                break

            second_point = cut_edge_points[second_point_index]

            if second_point[0] != first_point[0]:
                lin_a = (second_point[1] - first_point[1]) / (second_point[0] - first_point[0])
            else:
                lin_a = 1.0
            lin_b = first_point[1] - (lin_a * first_point[0])
            y = lin_a * x + lin_b
            points.append((int(round(x)), int(round(y))))
            if len(points) >= point_count:
                break

        return points

    def __generate_cut_edge(self) -> List[Tuple[int, int]]:
        cut_ends = set()
        for i in self.__sides[0] + self.__sides[1]:
            if i not in self.__base_points:
                cut_ends.add(self.__approx[i])

        start_adding: bool = False
        cut_side: List[Tuple[int, int]] = []

        for point in self.__contour:
            if start_adding:
                cut_side.append(point)

            if point in cut_ends and start_adding:
                break
            elif point in cut_ends:
                start_adding = True

        return cut_side

    def __generate_cut_distances(self, normalized_cut_points) -> np.array:
        base_point_1 = self.__contour[self.__base_points[0]]
        base_point_2 = self.__contour[self.__base_points[1]]

        distance_matches = np.zeros(shape=(50,), dtype=np.int16)
        for i, cut_point in enumerate(normalized_cut_points):
            average_base_y = round(np.average([base_point_1[1], base_point_2[1]]))
            distance = average_base_y - cut_point[1]
            distance_matches[i] = distance

        return distance_matches


class FragmentMetadataProvider:
    @staticmethod
    def get_metadata(image: np.ndarray, normalized_cut_point_count: int = 50) -> FragmentMetadata:
        contour = find_contour(image)

        epsilon = 0.005 * cv.arcLength(contour, closed=True)
        approx = cv.approxPolyDP(contour, epsilon, closed=True)

        base = FragmentMetadataProvider.__get_base_indices(approx)
        sides = FragmentMetadataProvider.__get_sides(base, len(approx))

        metadata = FragmentMetadata(base, sides, image, contour, approx, normalized_cut_point_count)

        return metadata

    @staticmethod
    def __get_base_indices(contour) -> Tuple[int, int]:
        result: tuple = (0, 1)
        result_length: float = 0.0

        for j in range(len(contour)):
            length: float = calculate_distance(contour[j][0], contour[(j + 1) % len(contour)][0])

            if length > result_length:
                result = j, (j + 1) % len(contour)
                result_length = length

        result = tuple(sorted(result))
        return result

    @staticmethod
    def __get_sides(base, len_contour) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        b_1: int = base[0]
        b_2: int = base[1]

        side_1: tuple = ((b_1 - 1) % len_contour, b_1) if (b_2 - b_1 == 1) else ((b_1 + 1) % len_contour, b_1)
        side_2: tuple = (b_2, (b_2 + 1) % len_contour) if (b_2 - b_1 == 1) else (b_2, (b_2 - 1) % len_contour)

        side_1 = tuple(sorted(side_1))
        side_2 = tuple(sorted(side_2))

        return side_1, side_2
