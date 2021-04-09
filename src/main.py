import math
import os
from typing import List, Dict

import cv2 as cv
import numpy as np


def calculate_length(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Copy image and padding
    res_img = img.copy()
    res_img = cv.copyMakeBorder(res_img, 64, 64, 64, 64, cv.BORDER_CONSTANT)

    # Get shape contours
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygon
    epsilon = 0.005 * cv.arcLength(contours[0], closed=True)
    approx = cv.approxPolyDP(contours[0], epsilon, closed=True)

    # Find longest line and straighten image
    start_point, end_point = get_longest_line_points(approx)
    angle = math.atan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0])) * (180.0 / math.pi)
    angle %= 180.0

    moments = cv.moments(contours[0])
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    rot_mat = cv.getRotationMatrix2D((center_y, center_x), angle, 1.0)

    height, width = res_img.shape
    res_img = cv.warpAffine(res_img, rot_mat, (width, height), flags=cv.INTER_LINEAR)

    # Cut black padding from image
    contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    x, y, w, h = cv.boundingRect(contours)
    res_img = res_img[y:y + h, x:x + w]

    # Scale image so that at least one dimension is 512px and other lesser that 512px
    img_h, img_w = res_img.shape
    scale_x = img_w / 512
    scale_y = img_h / 512

    scale = scale_y if scale_x < scale_y else scale_x

    new_width = round(img_w / scale)
    new_height = round(img_h / scale)
    res_img = cv.resize(res_img, (new_width, new_height))

    # Threshold image after resize (removes intermediate colors)
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)

    return res_img


def get_longest_line_points(contour):
    """ Finds longest line in contour

    :param contour:
    :return: Tuple with start and end point of line
    """
    result = (0, 0), (0, 0)
    result_length = 0.0

    for j in range(len(contour) - 1):
        length = calculate_length(contour[j][0], contour[j + 1][0])
        if length > result_length:
            result = contour[j][0], contour[j + 1][0]
            result_length = length

    return tuple(result[0]), tuple(result[1])


def approximate_contours(contours):
    epsilon = 0.005 * cv.arcLength(contours[0], closed=True)
    approx = cv.approxPolyDP(contours[0], epsilon, closed=True)
    return approx


class Dataset:
    SHOW_IMAGES_ON_LOAD = False
    SHOW_IMAGES = True
    correct: dict
    images: List[np.ndarray]

    def __init__(self):
        self.correct = {}
        self.images = []
        self.edge_lengths: Dict[int, float] = {}

    def solve(self):
        print()
        approx_vertices: Dict[int, list] = {}
        images_area: Dict[int, np.int32] = {}

        for i, image in enumerate(self.images):
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            approx = approximate_contours(contours)

            # TODO: Decide which area is better
            # area = cv.contourArea(approx)
            area = np.sum(image, dtype=np.int32)
            images_area[i] = area

            if len(approx) not in approx_vertices.keys():
                approx_vertices[len(approx)] = [i]
            else:
                approx_vertices[len(approx)] += [i]

            if Dataset.SHOW_IMAGES:
                img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
                cv.drawContours(img_copy, [approx], 0, color=(0, 0, 255), thickness=2)
                cv.imshow("test", img_copy)
                cv.waitKey()

        result = {}
        expected_area = np.int32(448 * 448 * 255)
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)):
                joint_area = images_area[i] + images_area[j]
                deviance = np.abs(expected_area - joint_area)
                if deviance < 100000:
                    print(f"{i}-{j} : {deviance}")

        # TODO: Find best way to match rectangle fragments
        print(f"Vertices: {approx_vertices}")
        print(f"Area: {images_area}")
        # print()
        # print(self.edge_lengths)
        # print()
        # print(images_points)
        # result = {}
        # remaining = []
        # for elem in approx_points.values():
        #     if len(elem) == 2:
        #         result[elem[0]] = elem[1]
        #         result[elem[1]] = elem[0]
        #     elif len(elem) == 1:
        #         remaining.append(elem[0])
        #
        # if len(remaining) == 2:
        #     result[remaining[0]] = remaining[1]
        #     result[remaining[1]] = remaining[0]

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
