import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src1",
        help="The first file with sources",
        type=Path,
    )
    parser.add_argument(
        "src2",
        help="The second file with sources",
        type=Path,
    )
    parser.add_argument(
        "--tgt1",
        help="The first file with targets",
        type=Path,
    )
    parser.add_argument(
        "--tgt2",
        help="The second file with targets",
        type=Path,
    )
    args = parser.parse_args()
    args.src1 = args.src1.expanduser()
    args.src2 = args.src2.expanduser()
    if args.tgt1 is not None:
        args.tgt1 = args.tgt1.expanduser()
    if args.tgt2 is not None:
        args.tgt2 = args.tgt2.expanduser()
    return args


def main():
    args = get_args()
    if args.tgt1 is not None or args.tgt2 is not None:
        with args.src1.open() as src1, \
                args.src2.open() as src2, \
                args.tgt1.open() as tgt1, \
                args.tgt2.open() as tgt2:
            s1 = set(zip(src1.readlines(), tgt1.readlines()))
            s2 = set(zip(src2.readlines(), tgt2.readlines()))
            print(f"Number of duplicates between ({args.src1}, {args.tgt1}) and ({args.src2}, {args.tgt2}):",
                  len(s1 & s2))
    else:
        with args.src1.open() as src1, \
                args.src2.open() as src2:
            s1 = set(src1.readlines())
            s2 = set(src2.readlines())
        print(f"Number of duplicates between {args.src1} and {args.src2}:", len(s1 & s2))


if __name__ == "__main__":
    main()
