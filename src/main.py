from typing import List

import numpy as np

from src.dataset import Dataset
from src.solvers import DatasetSolver, DistanceToBaseDatasetSolver


def get_datasets_scores(solver: DatasetSolver, datasets: List[Dataset]) -> List[float]:
    scores: List[float] = list()
    for dataset in datasets:
        all_matches = solver.solve(dataset)
        score: float = 0.0
        for matches in all_matches:
            for i, match in enumerate(matches[1]):
                if match == dataset.correct[matches[0]]:
                    score += 1 / (i + 1)
        score /= len(dataset.correct)
        scores.append(round(score, 4))

    print(scores)
    print(round(np.average(scores), 4))

    return scores


def main():
    Dataset.VERBOSE = False
    Dataset.SHOW_IMAGES_ON_LOAD = False

    dataset_name: str = "A"
    dataset_size: int = 9
    dataset_start_index: int = 0

    set_range = range(dataset_start_index, dataset_start_index + dataset_size)
    datasets: List[Dataset] = [Dataset.load(f"./datasets/{dataset_name}/set{i}/") for i in set_range]

    get_datasets_scores(DistanceToBaseDatasetSolver(), datasets)


if __name__ == '__main__':
    main()
