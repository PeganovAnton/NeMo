import argparse
import json
from collections import Counter
from itertools import zip_longest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--originals_file",
        required=True,
        help="Path to txt file with originals."
    )
    parser.add_argument(
        "--translations_file",
        required=True,
        help="Path to file with translations."
    )
    parser.add_argument(
        "--stats_output",
        "-o",
        help="Path to output json file with stats."
    )
    args = parser.parse_args()
    return args


def compute_stats(lines):
    lengths = [len(l) for l in lines]
    return dict(Counter(lengths))


def main():
    args = get_args()
    original_lines = []
    translation_lines = []
    pair_lines = []
    with open(args.originals_file) as of, open(args.translations_file) as tf:
        for o_line, t_line in zip_longest(of, tf):
            o_line = o_line.strip()
            t_line = t_line.strip()
            original_lines.append(o_line)
            translation_lines.append(t_line)
            pair_lines.append(o_line + t_line)
    result = {
        "original_stats": compute_stats(original_lines),
        "translation_stats": compute_stats(translation_lines),
        "pair_stats": compute_stats(pair_lines),
    }
    with open(args.stats_output, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
