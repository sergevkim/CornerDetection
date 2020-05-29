import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-filename',
        default="{}/data/serge.jpg".format(Path.cwd()),
        type=str,
        help="image filename, default: ./data/serge.jpg"
    )

    return parser.parse_args()

