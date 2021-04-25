from typing import List

import numpy as np

from src.solvers import DatasetSolver, DistanceToBaseDatasetSolver


def get_datasets_scores(datasets: List[DatasetSolver]) -> List[float]:
    scores: List[float] = list()
    for dataset in datasets:
        scores.append(dataset.solve())

    print(scores)
    print(round(np.average(scores), 4))

    return scores


def main():
    # dataset_name: str = "A"
    # dataset_size: int = 9
    # dataset_start_index: int = 0
    dataset_name: str = "A-temp"
    dataset_size: int = 1
    dataset_start_index: int = 8

    set_range = range(dataset_start_index, dataset_start_index + dataset_size)
    datasets: List[DatasetSolver] = \
        [DistanceToBaseDatasetSolver.load(f"./datasets/{dataset_name}/set{i}/") for i in set_range]

    get_datasets_scores(datasets)


if __name__ == '__main__':
    main()
