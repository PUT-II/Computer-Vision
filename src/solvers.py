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


class AngleGradientDatasetSolver(DatasetSolver):
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

        grade_matches: Dict[int, Dict[int, float]] = {}
        for i in range(len(self.images) - 1):
            grade_matches[i] = {}
            gradient_1 = metadata_list[i].gradient
            for j in range(i + 1, len(self.images)):
                gradient_2 = metadata_list[j].gradient
                grade_matches[i][j] = abs(gradient_1 + gradient_2)

        result = {}
        for match in grade_matches:
            match_dict = grade_matches[match]

            match_keys = sorted(match_dict, key=match_dict.get)
            for key in match_keys:
                # if abs(metadata_1.normalized_cut_length - metadata_2.normalized_cut_length) > 0.06:
                #     continue

                if key not in result and match not in result:
                    result[match] = key
                    result[key] = match
                    break

        if self.VERBOSE:
            print(json.dumps(grade_matches, indent=1))
            print(f"Correct : {self.correct}")
            print(f"Result : {result}")
            print(grade_matches[1])
            print(grade_matches[4])

        score = 0
        for result_key, correct_key in zip(sorted(result), self.correct):
            if result[result_key] == self.correct[correct_key]:
                score += 1
        score /= len(self.correct)

        return round(score, 4)

    @staticmethod
    def load(directory: str) -> DatasetSolver:
        return DatasetSolver._load_instance(AngleGradientDatasetSolver(), directory)
