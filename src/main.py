from typing import List

import numpy as np

from src.dataset import Dataset


def get_datasets_scores(datasets: List[Dataset]):
    scores = list()
    # datasets = [datasets[1]]
    # datasets = datasets[1:]
    for dataset in datasets:
        scores.append(dataset.solve())

    print(scores)
    print(round(np.average(scores), 4))

    return scores


def main():
    datasets: List[Dataset] = [Dataset.load(f"./datasets/A/set{i}/") for i in range(8 + 1)]

    get_datasets_scores(datasets)


if __name__ == '__main__':
    main()
