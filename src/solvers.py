import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from src.dataset import Dataset
from src.fragment import FragmentMetadata
from src.fragment import FragmentMetadataProvider


class DatasetSolver(ABC):
    VERBOSE = False

    @abstractmethod
    def solve(self, dataset: Dataset) -> List[Tuple[int, List[int]]]:
        pass


class DistanceToBaseDatasetSolver(DatasetSolver):

    def solve(self, dataset: Dataset) -> List[Tuple[int, List[int]]]:
        metadata_list: List[FragmentMetadata] = []
        for i, image in enumerate(dataset.images):
            metadata = FragmentMetadataProvider.get_metadata(image)
            metadata_list.append(metadata)

        distance_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(len(metadata_list))}
        for i in range(len(metadata_list) - 1):
            distances_1 = metadata_list[i].distance_to_base_array

            for j in range(len(metadata_list)):
                if i == j:
                    continue

                distances_2 = metadata_list[j].distance_to_base_array
                distance_sum = distances_1 + np.flip(distances_2)
                average = np.average(distance_sum)
                score_array = 1 - np.abs(distance_sum - average) / average
                score = np.average(score_array)

                distance_scores[i][j] = score
                distance_scores[j][i] = score

        result: List[Tuple[int, List[int]]] = []
        for key in distance_scores:
            score_dict = distance_scores[key]

            match_keys = sorted(score_dict, key=score_dict.get, reverse=True)
            result.append((key, match_keys))

        if DatasetSolver.VERBOSE:
            for metadata in metadata_list:
                metadata.draw_features()

            print(json.dumps(distance_scores, indent=1))
            print(f"Correct : {dataset.correct}")
            print(f"Result : {result}")

        return result

    @staticmethod
    def __distance_to_base(base_point_1, base_point_2, cut_edge_point) -> int:
        average_base_y = round(np.average([base_point_1[1], base_point_2[1]]))
        return average_base_y - cut_edge_point[1]
