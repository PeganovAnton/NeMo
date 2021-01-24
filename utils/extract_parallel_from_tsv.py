import argparse
from pathlib import Path

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name",
        help="The name of the file which will be parsed.",
        type=Path,
    )
    parser.add_argument(
        "col1",
        help="The number of the column in tsv file which contains sentences.",
        type=int,
    )
    parser.add_argument(
        "output1",
        help="The name of the file which will contain the sentences from `col1`.",
        type=Path,
    )
    parser.add_argument(
        "col2",
        help="The number of the column in tsv file which contains sentences.",
        type=int,
    )
    parser.add_argument(
        "output2",
        help="The name of the file which will contain the sentences from `col2`.",
        type=Path,
    )
    args = parser.parse_args()
    args.file_name = args.file_name.expanduser()
    args.output1 = args.output1.expanduser()
    args.output2 = args.output2.expanduser()
    return args


def main():
    args =  get_args()
    df = pd.read_csv(args.file_name, header=None, dtype=str, sep="\t")
    df.to_csv(args.output1, columns=[f"X{args.col1}"], header=None, index=None)
    df.to_csv(args.output2, columns=[f"X{args.col2}"], header=None, index=None)


if __name__ == "__main__":
    main()
