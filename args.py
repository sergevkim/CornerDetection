import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-filename',
        default="{}/data/basket/serge.jpg".format(Path.cwd()),
        type=str,
        help="image filename, default: ./data/basket/serge.jpg")
    parser.add_argument(
        '--symbols-dir',
        default="{}/data/symbols".format(Path.cwd()),
        type=str,
        help="image filename, default: ./data/symbols")

    return parser.parse_args()

