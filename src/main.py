import os
from typing import List, Dict

import cv2 as cv
import numpy as np


class Dataset:
    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []

    def solve(self):
        images_points: Dict[int, list] = {}
        for i, image in enumerate(self.images):
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            epsilon = 0.015 * cv.arcLength(contours[0], True)
            approx = cv.approxPolyDP(contours[0], epsilon, True)
            if len(approx) not in images_points.keys():
                images_points[len(approx)] = [i]
            else:
                images_points[len(approx)] += [i]

            img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
            cv.drawContours(img_copy, [approx], 0, color=(0, 0, 255), thickness=3)
            cv.imshow("test", img_copy)
            cv.waitKey()
        print(images_points)
        # for i in images_points:


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
