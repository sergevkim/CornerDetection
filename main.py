from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

import cv2

from args import _parse_args


ALPHABET = [chr(i) for i in range(32, 127)]
TILE_SIZE = (7, 7)


class Tile:
    def __init__(self, image, name=None):
        self.image = image
        self.name = name

    def __str__(self):
        n, m = self.image.shape
        result = []

        for i in range(n):
            for j in range(m):
                if self.image[i][j] < 127:
                    result.append('.')
                else:
                    result.append('#')
            result.append('\n')

        result.pop()

        return ''.join(result)

    def handle(self):
        '''
        SIFT, SURF transforms
        '''
        pass


def handle(image):
    n, m = image.shape
    result = [[None for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            if image[i][j] < 127:
                result[i][j] = 255
            else:
                result[i][j] = 0
        #result.append('\n')

    #result.pop()

    return np.array(result)


def similar_coef(tile_1, tile_2):
    mse = mean_squared_error(tile_1.image, tile_2.image)
    return mse


def find_best_symbol(tile, symbols):
    '''
    A way to find the best symbol from the symbols list
    '''
    best_coef = similar_coef(tile, symbols['!'])
    best_symbol = symbols['!']

    for i, symbol_name in enumerate(symbols):
        symbol = symbols[symbol_name]
        cur_coef = similar_coef(tile, symbol)
        if cur_coef < best_coef:
            best_coef = cur_coef
            best_symbol = symbol

    return best_symbol


def tiling(big_image, tile_size=TILE_SIZE):
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
        symbol_image = cv2.imread(str(image_filename), 0)
        symbol = Tile(symbol_image, name=symbol_name)
        symbols[symbol_name] = symbol

    return symbols


def main():
    args = _parse_args()

    symbols = read_symbols(args.symbols_dir)

    big_image = cv2.imread(args.image_filename, 0) #as grayscale

    big_image = handle(big_image)

    tiles = tiling(big_image) # list of lists
    n = len(tiles)
    m = len(tiles[0])

    print('---')
    print(tiles[0][0])
    print('---')
    for s in symbols:
        print(s, similar_coef(symbols[s], tiles[0][0]))

    result_text = [[None for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            ascii_symbol = find_best_symbol(tiles[i][j], symbols)
            result_text[i][j] = ascii_symbol

    print(type(result_text[0][0].image))
    #'''
    for i in range(n):
        print(Tile(np.concatenate([result_text[i][j].image for j in range(m)], axis=1)))
    '''
    for i in range(n):
        print(Tile(np.concatenate([tiles[i][j].image for j in range(m)], axis=1)))
    '''
if __name__ == "__main__":
    main()

