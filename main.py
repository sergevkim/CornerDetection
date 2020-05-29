import cv2
import random

from args import _parse_args


ALPHABET = [chr(i) for i in range(32, 127)]


def symbol_match(image):
    '''
    A way to find the best symbol
    '''
    symbol = ALPHABET[random.randint(1, 30)]
    return symbol


def handle(image, extended=False):
    '''
    We should handle the image to find the best symbol to it
    '''
    return symbol_match(image)


def tiling(big_image):
    '''
    Cut the big image for a list of lists of small tiles
    '''
    return [[0 for i in range(4)] for j in range(6)]


def main():
    args = _parse_args()

    big_image = cv2.imread(args.image_filename)

    images = tiling(big_image) # list of lists
    n = len(images)
    m = len(images[0])

    result_text = [[None for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            ascii_symbol = handle(images[i][j], extended=False)
            result_text[i][j] = ascii_symbol

    print('\n'.join(''.join(result_text)))


if __name__ == "__main__":
    main()

