import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to the file in which separators will be balanced.",
        type=Path,
    )
    parser.add_argument(
        "output_file",
        help="Path to the output file.",
        type=Path,
    )
    parser.add_argument(
        "sep",
        help="A character which serves as a separator."
    )
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main():
    args = get_args()
    max_num_sep = 0
    with args.input_file.open() as f:
        for l in f:
            n = l.count(args.sep)
            if n > max_num_sep:
                max_num_sep = n
    with args.input_file.open() as in_f, args.output_file.open('w') as out_f:
        for l in in_f:
            out_f.write(l.rstrip('\n') + args.sep * (max_num_sep - l.count(args.sep)) + '\n')


if __name__ == "__main__":
    main()
