import copy
from pathlib import Path

import numpy as np
import cv2
from sklearn.metrics import mean_squared_error


TILE_SIZE = (7, 7)


class Tile:
    def __init__(self, image, name=None):
        self.image = image / 255.
        self.area = np.size(image)
        self.name = name
        self.hcd = self.harris_corner_detector()
        self.skel = self.skeleton()
        self.sift = self.scale_invariant_feature_tranform()

    def __str__(self):
        n, m = self.image.shape
        result = []

        for i in range(n):
            for j in range(m):
                if self.image[i][j] < 64:
                    result.append('.')
                else:
                    result.append('#')
            result.append('\n')

        result.pop()

        return ''.join(result)

    def harris_corner_detector(self):
        cv2.imwrite("./data/tmp.jpg", self.image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        img = cv2.imread("./data/tmp.jpg", flags=cv2.IMREAD_GRAYSCALE)
        img = np.abs(img)
        result = cv2.cornerHarris(img, 1, 1, 0.01)
        norm = np.linalg.norm(result)
        if norm != 0:
            result /= norm

        return result

    def skeleton(self):
        cv2.imwrite("./data/tmp.jpg", self.image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        img = cv2.imread("./data/tmp.jpg", flags=cv2.IMREAD_GRAYSCALE)
        skel = np.zeros_like(img)
        size = np.size(img)

        _, img = cv2.threshold(img, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, element)
            tmp = cv2.dilate(eroded, element)
            tmp = cv2.subtract(img, tmp)
            skel = cv2.bitwise_or(skel, tmp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                break

        return skel

    def scale_invariant_feature_tranform(self):
        cv2.imwrite("./data/tmp.jpg", self.image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        img = cv2.imread("./data/tmp.jpg", flags=cv2.IMREAD_GRAYSCALE)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

        return kp


def similar_coef(tile_1, tile_2, mode):
    if mode == 'mse':
        mse_i = mean_squared_error(tile_1.image, tile_2.image)
        mse_h = mean_squared_error(tile_1.hcd, tile_2.hcd)
        #mse_skel = mean_squared_error(tile_1.skel, tile_2.skel)
        result = mse_i + mse_h * 4
        result /= 5

        return result

    elif mode == 'area':
        area_i = abs(np.sum(tile_1.image) - np.sum(tile_2.image))
        mse_h = mean_squared_error(tile_1.hcd, tile_2.hcd)
        result = area_i + mse_h
        result /= 2

        return result

    elif mode == 'mixed':
        area_i = abs(np.sum(tile_1.image) - np.sum(tile_2.image))
        mse_i = mean_squared_error(tile_1.image, tile_2.image)
        mse_h = mean_squared_error(tile_1.hcd, tile_2.hcd)
        result = area_i / 10 + mse_i + mse_h * 3

        return result


def prepare_image(image):
    n, m = image.shape
    image = cv2.resize(image, (m * 2, n), interpolation=cv2.INTER_AREA)
    result = np.array(list(map(lambda row: list(map(lambda x: int(255 if x >= 248 else x), row)), image)))

    return result


def tile_image(big_image, tile_size=TILE_SIZE):
    '''
    Cut the big image for a list of lists of small tiles
    '''
    n = big_image.shape[0] // tile_size[0]
    m = big_image.shape[1] // tile_size[1]

    tiles = [[None for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            y_1 = tile_size[0] * i
            y_2 = tile_size[0] * (i + 1)
            x_1 = tile_size[1] * j
            x_2 = tile_size[1] * (j + 1)
            tiles[i][j] = Tile(big_image[y_1:y_2, x_1:x_2])

    return tiles


def read_symbols(dir_name):
    symbols = {}

    dir_path = Path(dir_name)

    for image_filename in dir_path.glob("*.jpg"):
        symbol_name = image_filename.stem
        if symbol_name == "point":
            symbol_name = '.'
        elif symbol_name == "slash":
            symbol_name = '/'
        symbol_image = cv2.imread(str(image_filename), 0)
        symbol = Tile(symbol_image, name=symbol_name)
        symbols[symbol_name] = symbol

    return symbols


def find_best_symbol(tile, symbols, mode):
    '''
    A way to find the best symbol from the symbols list
    '''
    best_coef = similar_coef(tile, symbols['!'], mode)
    best_symbol = symbols['!']

    for i, symbol_name in enumerate(symbols):
        symbol = symbols[symbol_name]
        cur_coef = similar_coef(tile, symbol, mode)
        if cur_coef < best_coef:
            best_coef = cur_coef
            best_symbol = symbol

    return best_symbol

