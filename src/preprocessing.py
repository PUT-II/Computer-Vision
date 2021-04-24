import math

import cv2 as cv
import numpy as np

from src.utils import get_longest_line_points


def preprocess_image(image: np.ndarray) -> np.ndarray:
    res_img = image.copy()

    # Rescale image
    # res_img = __scale_image(res_img, 512)

    res_img = __add_padding(res_img, 256)

    # Get shape contours
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    if len(contours) >= 4:
        res_img = __rotate_image(res_img, contours)

    # Scale image so that at least one dimension is 512px and other lesser that 512px
    # res_img = __scale_image(res_img, 512)

    # Threshold image after preprocessing operations
    _, res_img = cv.threshold(res_img, 127, 255, cv.THRESH_BINARY)
    res_img = __add_padding(res_img, 32)

    return res_img


def __rotate_image(image: np.ndarray, contours: list) -> np.ndarray:
    res_img = image.copy()

    # Approximate contours to polygon
    epsilon = 0.005 * cv.arcLength(contours, closed=True)
    approx = cv.approxPolyDP(contours, epsilon, closed=True)

    # Find longest line and straighten image
    start_point, end_point = get_longest_line_points(approx)
    angle = math.atan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0])) * (180.0 / math.pi)
    angle %= 180.0

    moments = cv.moments(contours)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        rot_mat = cv.getRotationMatrix2D((center_y, center_x), angle, 1.0)

        height, width = res_img.shape
        res_img = cv.warpAffine(res_img, rot_mat, (width, height), flags=cv.INTER_LINEAR)

        # Cut black padding from image
        res_img = __cut_empty_padding(res_img)

        contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0]

        # Approximate contours to polygon
        epsilon = 0.005 * cv.arcLength(contours, closed=True)
        approx = cv.approxPolyDP(contours, epsilon, closed=True)
        start_point, end_point = get_longest_line_points(approx)

        if abs(start_point[1] - end_point[1]) / res_img.shape[0] > 0.2:
            res_img = cv.copyMakeBorder(res_img, 256, 256, 256, 256, cv.BORDER_CONSTANT)
            height, width = res_img.shape
            rot_mat = cv.getRotationMatrix2D((round(height / 2), round(width / 2)), 90.0, 1.0)
            res_img = cv.warpAffine(res_img, rot_mat, (width, height), flags=cv.INTER_LINEAR)

            # Cut black padding from image
            res_img = __cut_empty_padding(res_img)
    return res_img


def __add_padding(image: np.ndarray, pad_size: int) -> np.ndarray:
    return cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT)


def __cut_empty_padding(image: np.ndarray) -> np.ndarray:
    res_img = image.copy()

    # Cut black padding from image
    contours, _ = cv.findContours(res_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    x, y, w, h = cv.boundingRect(contours)
    res_img = res_img[y:y + h, x:x + w]
    return res_img


def __scale_image(image: np.ndarray, max_size) -> np.ndarray:
    res_img = image.copy()
    img_h, img_w = res_img.shape
    scale_x = img_w / max_size
    scale_y = img_h / max_size

    scale = scale_y if scale_x < scale_y else scale_x

    new_width = round(img_w / scale)
    new_height = round(img_h / scale)
    res_img = cv.resize(res_img, (new_width, new_height))
    return res_img
