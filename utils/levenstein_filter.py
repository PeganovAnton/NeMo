import argparse
from pathlib import Path

from Levenshtein import distance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_src",
        help="The path to the input file with translation source.",
        type=Path,
    )
    parser.add_argument(
        "input_tgt",
        help="The path to the input file with translation target.",
        type=Path,
    )
    parser.add_argument(
        "output_src",
        help="The path to the output file with translation source.",
        type=Path,
    )
    parser.add_argument(
        "output_tgt",
        help="The path to the output file with translation target.",
        type=Path,
    )
    parser.add_argument(
        "--threshold",
        "-t",
        help="The minimum allowed ratio of levenshtein distance between source and target to maximum of the lengths of "
             "source and target.",
        type=float,
        default=0.2,
    )
    args = parser.parse_args()
    args.input_src = args.src.expanduser()
    args.input_tgt = args.tgt.expanduser()
    args.output_src = args.output_src.expanduser()
    args.output_tgt = args.output_tgt.expanduser()
    return args


def main():
    args = get_args()
    with args.input_src.open() as isf, args.input_tgt.open() as itf, args.output_src.open() as osf, args.output_tgt.open() as otf:
        for i, (s, t) in enumerate(zip(isf, itf)):
            if distance(s, t) / max(len(s), len(t)) > args.threshold:
                osf.write(s.strip() + '\n')
                otf.write(t.strip() + '\n')




if __name__ == "__main__":
    main()
