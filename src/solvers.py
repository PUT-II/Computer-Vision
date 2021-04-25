import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import cv2 as cv
import numpy as np

from src.fragment import Fragment
from src.fragment import FragmentMetadata
from src.preprocessing import preprocess_image


class DatasetSolver(ABC):
    SHOW_IMAGES_ON_LOAD = False

    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []
        self.edge_lengths: Dict[int, float] = {}

    @abstractmethod
    def solve(self) -> float:
        pass

    @staticmethod
    def load(directory: str):
        pass

    @staticmethod
    def _load_instance(instance, directory: str):
        instance = instance

        with open(f"{directory}correct.txt", mode="r") as file:
            lines: List[str] = file.readlines()
        for i in range(len(lines)):
            instance.correct[i] = int(lines[i])

        filenames: List[str] = os.listdir(directory)
        filename_numbers = [int(filename.replace(".png", "")) for filename in filenames if filename != "correct.txt"]
        for filename_number in sorted(filename_numbers):
            filename = f"{filename_number}.png"
            file_path = f"{directory}{filename}"

            img: np.ndarray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            img = preprocess_image(img)

            if DatasetSolver.SHOW_IMAGES_ON_LOAD:
                print(file_path)
                cv.imshow("test", img)
                cv.waitKey()

            instance.images.append(img)

        return instance


class DistanceToBaseDatasetSolver(DatasetSolver):
    VERBOSE = False

    def solve(self) -> float:
        metadata_list: List[FragmentMetadata] = []
        for i, image in enumerate(self.images):
            metadata = Fragment.get_metadata(image)
            metadata_list.append(metadata)

        if self.VERBOSE:
            for i, metadata in enumerate(metadata_list):
                print(f"Fragment {i}")
                print(metadata.gradient)
                metadata.draw_features()

        distance_matches: Dict[int, np.array] = {}
        for i, metadata in enumerate(metadata_list):
            distance_matches[i] = np.zeros(shape=(50,), dtype=np.int16)

            base_point1 = metadata.contour[metadata.base_points[0]]
            base_point2 = metadata.contour[metadata.base_points[1]]

            for j, cut_point in enumerate(metadata.normalized_cut_points):
                distance = self.__distance_to_base(base_point1, base_point2, cut_point)
                distance_matches[i][j] = distance

        match_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(len(metadata_list))}
        for i in range(len(metadata_list) - 1):
            distances_1 = distance_matches[i]

            for j in range(i + 1, len(metadata_list)):
                distances_2 = distance_matches[j]
                distance_sum = distances_1 + np.flip(distances_2)
                average = np.average(distance_sum)
                score_array = 1 - np.abs(distance_sum - average) / average
                score = round(np.average(score_array), 4)

                match_scores[i][j] = score
                match_scores[j][i] = score

        result: Dict[int, int] = {}
        for dict_key in match_scores:
            score_dict = match_scores[dict_key]

            match_keys = sorted(score_dict, key=score_dict.get, reverse=True)
            for key in match_keys:
                if key not in result and dict_key not in result:
                    result[dict_key] = key
                    result[key] = dict_key
                    break

        if self.VERBOSE:
            print(json.dumps(match_scores, indent=1))
            print(f"Correct : {self.correct}")
            print(f"Result : {result}")

        score: float = 0.0
        for result_key, correct_key in zip(sorted(result), self.correct):
            if result[result_key] == self.correct[correct_key]:
                score += 1
        score /= len(self.correct)

        return round(score, 4)

    @staticmethod
    def load(directory: str) -> DatasetSolver:
        return DatasetSolver._load_instance(DistanceToBaseDatasetSolver(), directory)

    @staticmethod
    def __distance_to_base(base_point_1, base_point_2, cut_edge_point) -> int:
        average_base_y = round(np.average([base_point_1[1], base_point_2[1]]))
        return average_base_y - cut_edge_point[1]
