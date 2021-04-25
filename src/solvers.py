import json
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from src.dataset import Dataset
from src.fragment import Fragment
from src.fragment import FragmentMetadata


class DatasetSolver(ABC):
    @abstractmethod
    def solve(self, dataset: Dataset) -> float:
        pass


class DistanceToBaseDatasetSolver(DatasetSolver):
    VERBOSE = False

    def solve(self, dataset: Dataset) -> float:
        metadata_list: List[FragmentMetadata] = []
        for i, image in enumerate(dataset.images):
            metadata = Fragment.get_metadata(image)
            metadata_list.append(metadata)

        if self.VERBOSE:
            for i, metadata in enumerate(metadata_list):
                print(f"Fragment {i}")
                metadata.draw_features()

        match_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(len(metadata_list))}
        for i in range(len(metadata_list) - 1):
            distances_1 = metadata_list[i].distance_to_base_array

            for j in range(i + 1, len(metadata_list)):
                distances_2 = metadata_list[j].distance_to_base_array
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
            print(f"Correct : {dataset.correct}")
            print(f"Result : {result}")

        score: float = 0.0
        for result_key, correct_key in zip(sorted(result), dataset.correct):
            if result[result_key] == dataset.correct[correct_key]:
                score += 1
        score /= len(dataset.correct)

        return round(score, 4)

    @staticmethod
    def __distance_to_base(base_point_1, base_point_2, cut_edge_point) -> int:
        average_base_y = round(np.average([base_point_1[1], base_point_2[1]]))
        return average_base_y - cut_edge_point[1]
