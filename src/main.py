import os
from typing import List, Dict

import cv2 as cv
import numpy as np


def calculate_length(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def preprocess_image(img: np.ndarray) -> np.ndarray:
    res_img = img.copy()
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)

    return res_img


class Dataset:
    SHOW_IMAGES_ON_LOAD = False
    SHOW_IMAGES = True
    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []

    def solve(self):
        images_points: Dict[int, list] = {}
        edge_lengths: Dict[int, float] = {}

        for i, image in enumerate(self.images):
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            epsilon = 0.06 * cv.arcLength(contours[0], closed=True)
            approx = cv.approxPolyDP(contours[0], epsilon, closed=True)

            longest_edge = 0.0
            for j in range(len(approx) - 1):
                length = calculate_length(approx[j][0], approx[j + 1][0])
                if length > longest_edge:
                    longest_edge = length
            edge_lengths[i] = round(longest_edge, 3)

            if len(approx) not in images_points.keys():
                images_points[len(approx)] = [i]
            else:
                images_points[len(approx)] += [i]

            img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
            cv.drawContours(img_copy, [approx], 0, color=(0, 0, 255), thickness=2)
            if Dataset.SHOW_IMAGES:
                cv.imshow("test", img_copy)
                cv.waitKey()

        print()
        print(edge_lengths)
        print()
        # print(images_points)
        result = {}
        remaining = []
        for elem in images_points.values():
            if len(elem) == 2:
                result[elem[0]] = elem[1]
                result[elem[1]] = elem[0]
            elif len(elem) == 1:
                remaining.append(elem[0])

        if len(remaining) == 2:
            result[remaining[0]] = remaining[1]
            result[remaining[1]] = remaining[0]

        print(f"Correct : {self.correct}")
        print(f"Result : {result}")
        score = 0
        for result_key, correct_key in zip(sorted(result), self.correct):
            if result[result_key] != self.correct[correct_key]:
                score += 1
        score /= len(self.correct)
        print(f"Score : {score}")

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
                img: np.ndarray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                img = preprocess_image(img)

                if Dataset.SHOW_IMAGES_ON_LOAD:
                    print(file_path)
                    cv.imshow("test", img)
                    cv.waitKey()

                result.images.append(img)

        return result


def main():
    # dataset = Dataset.load(f"./datasets/A/set2/")
    # dataset.solve()

    for i in range(7):
        print(f"Dataset {i}", end="\t")
        dataset = Dataset.load(f"./datasets/A/set{i}/")
        dataset.solve()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
