from typing import List

import numpy as np

from src.solvers import AngleGradientDatasetSolver, DatasetSolver, DistanceToBaseDatasetSolver


def get_datasets_scores(datasets: List[DatasetSolver]):
    scores = list()
    # datasets = [datasets[1]]
    # datasets = datasets[1:]
    for dataset in datasets:
        scores.append(dataset.solve())

    print(scores)
    print(round(np.average(scores), 4))

    return scores


def main():
    datasets: List[DatasetSolver] = [DistanceToBaseDatasetSolver.load(f"./datasets/A/set{i}/") for i in range(8 + 1)]

    get_datasets_scores(datasets)


if __name__ == '__main__':
    main()
