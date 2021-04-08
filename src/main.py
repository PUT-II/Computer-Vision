import os
from typing import List

import cv2 as cv
import numpy as np


class Dataset:
    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []

    def solve(self):
        for image in self.images:
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            new_contours = []
            for cnt in contours:
                epsilon = 0.015 * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                new_contours.append(approx)

            img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
            cv.drawContours(img_copy, new_contours, 0, color=(0, 0, 255), thickness=3)
            cv.imshow("test", img_copy)
            cv.waitKey()

    @staticmethod
    def load(directory: str):
        result = Dataset()
        filenames: List[str] = os.listdir(directory)
        for filename in filenames:
            file_path = f"{directory}{filename}"
            if filename == "correct.txt":
                with open(file_path, mode="r") as file:
                    lines: List[str] = file.readlines()
                for i in range(len(lines)):
                    result.correct[i] = int(lines[i])
            else:
                result.images.append(cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2GRAY))
        return result


def main():
    dataset = Dataset.load("./datasets/A/set0/")
    dataset.solve()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
