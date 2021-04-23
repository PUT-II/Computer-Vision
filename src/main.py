import json
import math
import os
from typing import List, Dict

import cv2 as cv
import numpy as np


def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Copy image and padding
    res_img = img.copy()
    res_img = cv.copyMakeBorder(res_img, 256, 256, 256, 256, cv.BORDER_CONSTANT)

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
    res_img = cv.copyMakeBorder(res_img, 32, 32, 32, 32, cv.BORDER_CONSTANT)

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


def calculate_length(p_1, p_2):
    return ((p_1[0] - p_1[1]) ** 2 + (p_2[0] - p_2[1]) ** 2) ** 0.5


# Source: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def get_angles(contours) -> List[float]:
    angles = []
    for i in range(-len(contours) + 1, 0):
        p_1 = contours[i][0]
        p_2 = contours[i + 1][0]
        p_3 = contours[i + 2][0]

        angle = get_angle(p_1, p_2, p_3)
        angles.append(round(angle, 2))

    return angles


class Dataset:
    SHOW_IMAGES_ON_LOAD = False
    VERBOSE = False
    correct: dict
    images: List[np.ndarray]

    def __init__(self, approximation_precision: float = 0.05, penalty_wight: float = 1.0):
        self.correct = {}
        self.images = []
        self.edge_lengths: Dict[int, float] = {}
        self.approximation_precision = approximation_precision
        self.penalty_weight = penalty_wight

    def solve(self):
        angles = []
        for i, image in enumerate(self.images):
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            approx = self.approximate_contours(contours)

            angle = get_angles(approx)
            angles.append(angle)

            if Dataset.VERBOSE and i in (1, 4):
                img_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
                cv.drawContours(img_copy, [approx], 0, color=(0, 0, 255), thickness=2)
                cv.imshow(f"test_{i}", img_copy)

        if Dataset.VERBOSE:
            cv.waitKey()

        angle_matches: Dict[int, Dict[int, float]] = {}
        for i in range(len(self.images) - 1):
            angle_matches[i] = {}
            angles_1 = angles[i]
            for j in range(i + 1, len(self.images)):
                angles_2 = angles[j]
                angle_matches[i][j] = self.get_image_score(angles_1, angles_2)

        result = {}
        for match in angle_matches:
            match_dict = angle_matches[match]

            match_keys = sorted(match_dict, key=match_dict.get, reverse=True)
            for key in match_keys:
                if key not in result and match not in result:
                    result[match] = key
                    result[key] = match
                    break

        if Dataset.VERBOSE:
            print(json.dumps(angle_matches, indent=1))
            print(f"Correct : {self.correct}")
            print(f"Result : {result}")
            print(angle_matches[0])
            print(angle_matches[6])

        score = 0
        for result_key, correct_key in zip(sorted(result), self.correct):
            if result[result_key] == self.correct[correct_key]:
                score += 1
        score /= len(self.correct)

        return round(score, 4)

    def get_image_score(self, angles_1: list, angles_2: list) -> float:
        scores: List[float] = []
        matched_angles = set()

        # average_score = 1 - ((sum(angles_1) + sum(angles_2)) % 360) / 360

        for i in range(len(angles_1)):
            angle_1: float = angles_1[i]
            matched_angle_index: int = 0
            best_match_score: float = 0.0
            for j in range(len(angles_2)):
                if j in matched_angles:
                    continue

                angle_2 = angles_2[j]
                angle_sum = angle_1 + angle_2

                if angle_sum < 180.0 + 20.0:
                    score = 1 - abs(angle_sum - 180.0) / 180.0
                else:
                    score = 1 - abs(angle_sum - 360.0) / 360.0

                if score > best_match_score:
                    best_match_score = score
                    matched_angle_index = j

            scores.append(best_match_score)
            matched_angles.add(matched_angle_index)

        if scores:
            average_score = np.average(scores)
        else:
            average_score = 0.0

        len_1 = len(angles_1)
        len_2 = len(angles_2)
        penalty = 1.0 - np.abs(len_1 - len_2) / ((len_1 + len_2) / 2) * self.penalty_weight
        # penalty = 1.0
        return round(average_score * penalty, 4)
        # return round(average_score - penalty, 4)

    def approximate_contours(self, contours):
        epsilon = self.approximation_precision * cv.arcLength(contours[0], closed=True)
        approx = cv.approxPolyDP(contours[0], epsilon, closed=True)
        return approx

    @staticmethod
    def load(directory: str):
        result = Dataset()

        with open(f"{directory}correct.txt", mode="r") as file:
            lines: List[str] = file.readlines()
        for i in range(len(lines)):
            result.correct[i] = int(lines[i])

        filenames: List[str] = os.listdir(directory)
        filename_numbers = [int(filename.replace(".png", "")) for filename in filenames if filename != "correct.txt"]
        for filename_number in sorted(filename_numbers):
            filename = f"{filename_number}.png"
            file_path = f"{directory}{filename}"

            img: np.ndarray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            # try:
            #     img = preprocess_image(img)
            # except:
            #     pass

            if Dataset.SHOW_IMAGES_ON_LOAD:
                print(file_path)
                cv.imshow("test", img)
                cv.waitKey()

            result.images.append(img)

        return result


def find_best_approximation_precision(datasets):
    print("Best approximation precision")
    best_approximation_precision = 1.0
    best_score = 0.0

    from multiprocessing import Pool
    proc_pool = Pool(processes=os.cpu_count())

    for approximation_precision_int in range(15, 400, 5):
        # print(approximation_precision_int)
        approximation_precision = approximation_precision_int / 1000.0
        scores = list()
        proc_results = []
        for dataset in datasets:
            dataset.penalty_weight = BEST_PENALTY_WEIGHT
            dataset.approximation_precision = approximation_precision
            res = proc_pool.apply_async(dataset.solve)
            proc_results.append(res)

        for res in proc_results:
            scores.append(res.get())

        total_score = sum(scores) / len(scores)
        if total_score > best_score:
            best_approximation_precision = approximation_precision
            best_score = total_score

    proc_pool.close()
    return best_approximation_precision, best_score


def find_best_penalty_weight(datasets):
    print("Best penalty weight")
    best_penalty_weight = 1.0
    best_score = 0.0

    from multiprocessing import Pool
    proc_pool = Pool(processes=os.cpu_count())

    for penalty_weight_int in range(1, 2000, 10):
        # print(penalty_weight_int)
        penalty_weight = penalty_weight_int / 100.0
        scores = list()
        proc_results = []
        for dataset in datasets:
            dataset.approximation_precision = BEST_APPROXIMATION_PRECISION
            dataset.penalty_weight = penalty_weight
            res = proc_pool.apply_async(dataset.solve)
            proc_results.append(res)

        for res in proc_results:
            scores.append(res.get())

        total_score = sum(scores) / len(scores)
        if total_score > best_score:
            best_penalty_weight = penalty_weight
            best_score = total_score

    proc_pool.close()
    return best_penalty_weight, best_score


def get_datasets_scores(datasets):
    scores = list()
    # datasets = [datasets[1]]
    datasets = datasets[1:]
    for dataset in datasets:
        # dataset.approximation_precision = 0.025
        dataset.penalty_weight = BEST_PENALTY_WEIGHT
        dataset.approximation_precision = BEST_APPROXIMATION_PRECISION
        scores.append(dataset.solve())

    print(scores)
    print(round(np.average(scores), 4))

    return scores


BEST_APPROXIMATION_PRECISION = 0.015
BEST_PENALTY_WEIGHT = 8.41


def main():
    datasets = [Dataset.load(f"./datasets/A/set{i}/") for i in range(8 + 1)]

    # print(find_best_approximation_precision(datasets))
    # print(find_best_penalty_weight(datasets))
    get_datasets_scores(datasets)


if __name__ == '__main__':
    main()
