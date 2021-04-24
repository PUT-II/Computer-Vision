import math

import cv2 as cv
import numpy as np

from src.utils import get_longest_line_points, find_contour


def preprocess_image(image: np.ndarray) -> np.ndarray:
    res_img: np.ndarray = image.copy()

    # Rescale image
    res_img = __scale_image(res_img, 512)

    res_img = __add_padding(res_img, 256)

    # Get shape contours
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    contour = find_contour(res_img)

    if len(contour) >= 4:
        res_img = __rotate_image(res_img, contour)

    # Scale image so that at least one dimension is 512px and other lesser that 512px
    res_img = __scale_image(res_img, 512)

    # Threshold image after preprocessing operations
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    res_img = __add_padding(res_img, 32)

    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)

    return res_img


def __rotate_image(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    res_img = image.copy()

    # Approximate contours to polygon
    epsilon = 0.005 * cv.arcLength(contour, closed=True)
    approx = cv.approxPolyDP(contour, epsilon, closed=True)

    # Find longest line and straighten image
    start_point, end_point = get_longest_line_points(approx)
    angle = math.atan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0])) * (180.0 / math.pi)
    angle %= 180.0

    moments = cv.moments(contour)

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    rot_mat = cv.getRotationMatrix2D((center_y, center_x), angle, 1.0)

    height, width = res_img.shape
    res_img = cv.warpAffine(res_img, rot_mat, (width, height), flags=cv.INTER_LINEAR)

    # Cut black padding from image
    res_img = __cut_empty_padding(res_img)
    res_img = __rotate_side_to_bottom(res_img)

    return res_img


def __rotate_side_to_bottom(image: np.ndarray) -> np.ndarray:
    res_img: np.ndarray = image.copy()

    contour = find_contour(res_img)

    # Approximate contours to polygon
    epsilon = 0.005 * cv.arcLength(contour, closed=True)
    approx = cv.approxPolyDP(contour, epsilon, closed=True)
    start_point, end_point = get_longest_line_points(approx)

    base_correct: bool = abs(start_point[1] + end_point[1]) / (2 * res_img.shape[0]) > 0.7
    if base_correct:
        return res_img

    base_side: bool = abs(start_point[1] - end_point[1]) / res_img.shape[0] > 0.2
    base_left: bool = abs(start_point[0] - end_point[0]) / res_img.shape[1] > 0.2
    base_top: bool = abs(start_point[1] + end_point[1]) / (2 * res_img.shape[0]) < 0.2

    if base_top:
        rot_angle = 180.0

    elif base_side and base_left:
        rot_angle = 90.0
    elif base_side and base_left:
        rot_angle = -90.0
    else:
        rot_angle = 0.0

    if rot_angle != 0.0:
        res_img = cv.copyMakeBorder(res_img, 256, 256, 256, 256, cv.BORDER_CONSTANT)
        height, width = res_img.shape
        rot_mat = cv.getRotationMatrix2D((round(height / 2), round(width / 2)), rot_angle, 1.0)
        res_img = cv.warpAffine(res_img, rot_mat, (width, height), flags=cv.INTER_LINEAR)

        # Cut black padding from image
        res_img = __cut_empty_padding(res_img)
    return res_img


def __add_padding(image: np.ndarray, pad_size: int) -> np.ndarray:
    return cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT)


def __cut_empty_padding(image: np.ndarray) -> np.ndarray:
    res_img = image.copy()

    # Cut black padding from image
    contour = find_contour(image)

    x, y, w, h = cv.boundingRect(contour)
    res_img = res_img[y:y + h, x:x + w]
    return res_img


def __scale_image(image: np.ndarray, max_size: int) -> np.ndarray:
    res_img = image.copy()
    img_h, img_w = res_img.shape
    scale_x = img_w / max_size
    scale_y = img_h / max_size

    scale = scale_y if scale_x < scale_y else scale_x

    new_width = round(img_w / scale)
    new_height = round(img_h / scale)
    res_img = cv.resize(res_img, (new_width, new_height))
    return res_img
