import cv2

from src.args import _parse_args
from src.utils import Tile, prepare_image, tile_image, read_symbols, find_best_symbol


def main():
    args = _parse_args()

    symbols = read_symbols(args.symbols_dir)

    big_image = cv2.imread(args.image_filename, 0) #as grayscale

    if args.resize:
        big_image = cv2.resize(big_image, (420, 560)) #as grayscale

    big_image = prepare_image(big_image)
    print("prepared!")

    tiles = tile_image(big_image) # list of lists
    n = len(tiles)
    m = len(tiles[0])

    '''
    for s in symbols:
        print(symbols[s].name, similar_coef(symbols[s], tiles[0][0]))
    '''
    result_text = [[None for j in range(m)] for i in range(n)]

    print("go!")

    for i in range(n):
        for j in range(m):
            ascii_symbol = find_best_symbol(tiles[i][j], symbols, mode="mse")
            result_text[i][j] = ascii_symbol
        print(''.join([result_text[i][j].name for j in range(m)]))

    print(type(result_text[0][0].image))
    '''
    for i in range(n):
        print(Tile(np.concatenate([tiles[i][j].image for j in range(m)], axis=1)))
    '''
if __name__ == "__main__":
    main()

