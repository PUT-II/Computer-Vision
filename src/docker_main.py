import sys
from typing import List, Tuple

import numpy as np
from numpy import ndarray

from src.dataset import Dataset
from src.solvers import DistanceToBaseDatasetSolver


def main():
    set_path: str = sys.argv[1]
    image_count = int(sys.argv[2])

    dataset = Dataset.load(set_path, image_count=image_count, ignore_correct_file=True)

    result: List[Tuple[int, List[int]]] = DistanceToBaseDatasetSolver().solve(dataset)
    result_list: List[List[int]] = []
    for row in sorted(result, key=lambda elem: elem[0]):
        result_list.append(row[1])

    result_np: ndarray = np.array(result_list, dtype=int)
    # noinspection PyTypeChecker
    np.savetxt(sys.stdout, result_np, fmt=f'%d')


if __name__ == '__main__':
    main()
