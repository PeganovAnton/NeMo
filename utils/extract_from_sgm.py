import argparse
from pathlib import Path

import regex


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path
    )
    parser.add_argument(
        "output",
        type=Path,
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.input = args.input.expanduser()
    return args


def main():
    pattern = regex.compile(r'^<seg id="(?P<line_number>[1-9][0-9]*)">(?P<text>[^<])</seg>$')
    args = get_args()
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for line in in_f:
            m = pattern.match(line)
            if m is not None:
                out_f.write(m.groupdict()['text'])
                out_f.write('\n')


if __name__ == "__main__":
    main()
