import collections
from typing import Tuple, List

import cv2 as cv
import numpy as np

from src.utils import calculate_distance, find_contour


class FragmentMetadata:
    base_points: Tuple[int, int]
    sides: Tuple[Tuple[int, int], Tuple[int, int]]
    image: np.ndarray
    contour: List[Tuple[int, int]]
    approx: List[Tuple[int, int]]

    distance_to_base_array: np.array

    def __init__(self, base: Tuple[int, int], sides: tuple, image: np.ndarray, contour: np.ndarray, approx: np.ndarray,
                 cut_point_count: int = 50):
        self.base_points = base
        self.sides = sides

        self.approx = [tuple(point[0]) for point in approx]
        self.contour = [tuple(point[0]) for point in contour]
        self.contour = self.__shift_contour()

        self.image = image

        self.cut_edge = self.__generate_cut_edge()
        normalized_cut_points = self.__generate_normalized_cut_points(cut_point_count)
        self.distance_to_base_array = self.__generate_cut_distances(normalized_cut_points)

    def draw_features(self) -> None:
        image = cv.cvtColor(self.image.copy(), cv.COLOR_GRAY2BGR)

        if len(self.approx) < 4:
            return

        # Draw base
        base_start = self.approx[self.base_points[0]]
        base_end = self.approx[self.base_points[1]]
        image = cv.line(image, base_start, base_end, (0, 0, 255), 2)

        # Draw sides
        for i in range(len(self.sides)):
            side_start = self.approx[self.sides[i][0]]
            side_end = self.approx[self.sides[i][1]]
            image = cv.line(image, side_start, side_end, (255, 0, 255), 2)

        # Draw cut edge
        for i in range(len(self.cut_edge) - 1):
            point_1 = self.cut_edge[i]
            point_2 = self.cut_edge[i + 1]
            image = cv.line(image, point_1, point_2, (0, 255, 0), 2)

        cut_point_start = self.cut_edge[0]
        cut_point_end = self.cut_edge[len(self.cut_edge) - 1]
        image = cv.line(image, cut_point_start, cut_point_start, (0, 255, 0), 9)
        image = cv.line(image, cut_point_end, cut_point_end, (0, 255, 0), 9)

        cv.imshow("test", image)

        img_copy = cv.cvtColor(self.image.copy(), cv.COLOR_GRAY2BGR)
        for point in self.approx:
            img_copy = cv.line(img_copy, point, point, (0, 0, 255), 6)

        # Draw contour direction
        img_copy = cv.arrowedLine(img_copy, self.approx[0], self.approx[1], (0, 255, 255), 3)

        cut_ends = []
        for i in self.sides[0] + self.sides[1]:
            if i not in self.base_points:
                cut_ends.append(self.approx[i])
        for point in cut_ends:
            img_copy = cv.line(img_copy, point, point, (0, 255, 0), 20)

        cv.imshow("test-orig", img_copy)

        img_copy = cv.cvtColor(self.image.copy(), cv.COLOR_GRAY2BGR)
        for point in self.contour:
            img_copy = cv.line(img_copy, point, point, (0, 0, 255), 6)

        # Draw contour direction
        img_copy = cv.arrowedLine(img_copy, self.contour[0], self.approx[1], (0, 255, 255), 3)
        cv.imshow("test-orig-full", img_copy)

        cv.waitKey()

    def __shift_contour(self) -> List[Tuple[int, int]]:
        # Rotate approximated contour
        approx_index = self.approx.index(self.approx[self.base_points[0]])
        deque = collections.deque(self.approx)
        rot_val = -approx_index
        deque.rotate(rot_val)

        # Correct base and side edges points
        self.base_points = (
            (self.base_points[0] + rot_val) % len(self.approx), (self.base_points[1] + rot_val) % len(self.approx))
        side_1 = ((self.sides[0][0] + rot_val) % len(self.approx), (self.sides[0][1] + rot_val) % len(self.approx))
        side_2 = ((self.sides[1][0] + rot_val) % len(self.approx), (self.sides[1][1] + rot_val) % len(self.approx))
        self.sides = (side_1, side_2)
        self.approx = list(deque)

        # Rotate contour
        cnt_index = self.contour.index(self.approx[self.base_points[0]])
        deque = collections.deque(self.contour)
        deque.rotate(-cnt_index)
        return list(deque)

    def __generate_normalized_cut_points(self, point_count) -> List[Tuple[int, int]]:
        cut_edge_points = self.cut_edge
        cut_edge_points = list(sorted(cut_edge_points, key=lambda point: point[0]))

        cut_ends = []
        for i in self.sides[0] + self.sides[1]:
            if i not in self.base_points:
                cut_ends.append(self.approx[i])

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
        for i in self.sides[0] + self.sides[1]:
            if i not in self.base_points:
                cut_ends.add(self.approx[i])

        start_adding: bool = False
        cut_side: List[Tuple[int, int]] = []

        for point in self.contour:
            if start_adding:
                cut_side.append(point)

            if point in cut_ends and start_adding:
                break
            elif point in cut_ends:
                start_adding = True

        return cut_side

    def __generate_cut_distances(self, normalized_cut_points) -> np.array:
        base_point_1 = self.contour[self.base_points[0]]
        base_point_2 = self.contour[self.base_points[1]]

        distance_matches = np.zeros(shape=(50,), dtype=np.int16)
        for i, cut_point in enumerate(normalized_cut_points):
            average_base_y = round(np.average([base_point_1[1], base_point_2[1]]))
            distance = average_base_y - cut_point[1]
            distance_matches[i] = distance

        return distance_matches


class FragmentMetadataProvider:
    @staticmethod
    def get_metadata(image: np.ndarray, cut_point_count: int = 50) -> FragmentMetadata:
        contour = find_contour(image)

        epsilon = 0.005 * cv.arcLength(contour, closed=True)
        approx = cv.approxPolyDP(contour, epsilon, closed=True)

        base = FragmentMetadataProvider.__get_base_indices(approx)
        sides = FragmentMetadataProvider.__get_sides(base, len(approx))

        metadata = FragmentMetadata(base, sides, image, contour, approx, cut_point_count)

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
