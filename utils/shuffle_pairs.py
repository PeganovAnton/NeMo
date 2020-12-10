import argparse
import random
from itertools import zip_longest
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to file where files originals.txt and translations.txt are stored."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Path to the output dir. By default it is the same as input_dir."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. default is 42.",
        default=42,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    args.output_dir = args.output_dir.expanduser()
    args.input_dir = args.input_dir.expanduser()
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    input_originals_file = args.input_dir / Path("originals.txt")
    input_translations_file = args.input_dir / Path("translations.txt")
    originals, translations = [], []
    with input_originals_file.open() as of, input_translations_file.open() as tf:
        for o, t in zip_longest(of, tf):
            originals.append(o.strip())
            translations.append(t.strip())
    pairs = list(zip(originals, translations))
    random.shuffle(pairs)
    output_originals_file = args.output_dir / Path("originals.txt")
    output_translations_file = args.output_dir / Path("translations.txt")
    with output_originals_file.open('w') as of, output_translations_file.open('w') as tf:
        for o, t in pairs:
            of.write(o + '\n')
            tf.write(t + '\n')


if __name__ == "__main__":
    main()
