import os
from typing import Dict, List

import cv2 as cv
import numpy as np

from src.preprocessing import preprocess_image


class Dataset:
    __SHOW_IMAGES_ON_LOAD = False
    images: List[np.ndarray]
    correct: Dict[int, int]

    def __init__(self, images: List[np.ndarray], correct: Dict[int, int]):
        self.images = images
        self.correct = correct

    @staticmethod
    def load(directory: str):
        images: List[np.ndarray] = []
        correct: Dict[int, int] = {}

        with open(f"{directory}correct.txt", mode="r") as file:
            lines: List[str] = file.readlines()
        for i in range(len(lines)):
            correct[i] = int(lines[i])

        filenames: List[str] = os.listdir(directory)
        filename_numbers = [int(filename.replace(".png", "")) for filename in filenames if filename != "correct.txt"]
        for filename_number in sorted(filename_numbers):
            filename = f"{filename_number}.png"
            file_path = f"{directory}{filename}"

            img: np.ndarray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            img = preprocess_image(img)

            if Dataset.__SHOW_IMAGES_ON_LOAD:
                print(file_path)
                cv.imshow("test", img)
                cv.waitKey()

            images.append(img)

        return Dataset(images, correct)
