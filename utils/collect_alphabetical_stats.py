import argparse
from pathlib import Path

import homoglyphs as hg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to the input file.",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Path to the output text file with fractions.",
        type=Path,
    )
    parser.add_argument(
        "lang",
        help="The name of the expected language. Possible options are listed here https://github.com/life4/homoglyphs "
             "or with `homoglyphs.Languages.get_all()`."
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    alphabet = set(hg.Languages.get_alphabet([args.lang]))
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for l in in_f:
            count = 0
            for c in l:
                if c in alphabet:
                    count += 1
            out_f.write(str(count / len(l)) + '\n')


if __name__ == "__main__":
    main()
