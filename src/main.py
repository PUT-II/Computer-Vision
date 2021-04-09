import os
from typing import List, Dict

import cv2 as cv
import imutils
import numpy as np


def calculate_length(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def preprocess_image(img: np.ndarray) -> np.ndarray:
    res_img = img.copy()
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # gray = cv.GaussianBlur(res_img, (3, 3), 0)
    # mask = np.zeros(gray.shape, dtype="uint8")
    # cv.drawContours(mask, [contours[0]], -1, 255, -1)

    x, y, w, h = cv.boundingRect(contours[0])
    res_img = res_img[y:y + h, x:x + w]
    # maskROI = mask[y:y + h, x:x + w]
    # imageROI = cv.bitwise_and(res_img, res_img, mask=maskROI)
    # rot_img = imutils.rotate_bound(imageROI, 50)
    rot_img = res_img
    
    img_h, img_w = rot_img.shape
    scale_x = img_w / 496
    scale_y = img_h / 496

    scale = scale_y if scale_x < scale_y else scale_x

    new_width = round(img_w / scale)
    new_height = round(img_h / scale)
    rot_img = cv.resize(rot_img, (new_width, new_height))
    _, rot_img = cv.threshold(rot_img, 127, 255, cv.THRESH_BINARY)

    pad_x = round((512 - new_width) / 2)
    pad_y = round((512 - new_height) / 2)
    rot_img = cv.copyMakeBorder(rot_img, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT)
    rot_img = cv.resize(rot_img, (512, 512))

    return rot_img


def calculate_longest_dege(contour):
    longest_edge = 0.0
    for j in range(len(contour) - 1):
        length = calculate_length(contour[j][0], contour[j + 1][0])
        if length > longest_edge:
            longest_edge = length
    return longest_edge

class Dataset:
    SHOW_IMAGES_ON_LOAD = False
    SHOW_IMAGES = True
    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []
        self.edge_lengths: Dict[int, float] = {}

    def approx_fit(self, contours, i):
        epsilon = 0.005 * cv.arcLength(contours[0], closed=True)
        approx = cv.approxPolyDP(contours[0], epsilon, closed=True)
        return approx

    def solve(self):
        print()
        approx_points: Dict[int, list] = {}
        area_points: Dict[int, list] = {}

        for i, image in enumerate(self.images):
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            approx = self.approx_fit(contours, i)
            area = cv.contourArea(approx)

            if len(approx) not in approx_points.keys():
                approx_points[len(approx)] = [i]
            else:
                approx_points[len(approx)] += [i]

            area_points[i] = area

            img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
            cv.drawContours(img_copy, [approx], 0, color=(0, 0, 255), thickness=2)
            if Dataset.SHOW_IMAGES:
                cv.imshow("test", img_copy)
                cv.waitKey()
        print(f"Vertices: {approx_points}")
        print(f"Area_points: {area_points}")
        # print()
        # print(self.edge_lengths)
        # print()
        # print(images_points)
        result = {}
        remaining = []
        for elem in approx_points.values():
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
            if result[result_key] == self.correct[correct_key]:
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
    dataset = Dataset.load(f"./datasets/A/set1/")
    dataset.solve()

    # for i in range(7):
    #     print(f"Dataset {i}", end="\t")
    #     dataset = Dataset.load(f"./datasets/A/set{i}/")
    #     dataset.solve()
    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
