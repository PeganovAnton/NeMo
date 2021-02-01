import argparse
from pathlib import Path

from langdetect import detect


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_src",
        help="Path to the input source file.",
        type=Path,
    )
    parser.add_argument(
        "--input_tgt",
        help="Path to the input target file.",
        type=Path,
    )
    parser.add_argument(
        "output_src",
        help="Path to the file which will contain output source.",
        type=Path,
    )
    parser.add_argument(
        "--output_tgt",
        help="Path to the output target file",
        type=Path,
    )
    parser.add_argument(
        "source_lang",
        help="Input language. See https://github.com/Mimino666/langdetect."
    )
    parser.add_argument(
        "--target_lang",
        help="Output language. See https://github.com/Mimino666/langdetect."
    )
    args = parser.parse_args()
    if not (args.output_tgt is None and args.input_tgt is None and args.source_lang is None) \
            or not (args.output_tgt is not None and args.input_tgt is None and args.target_lang is not None):
        raise ValueError(
            f"Arguments `input_tgt`, `output_tgt`, `target_lang` have to be either `None` simultaneously or not `None`"
            f"simultaneously. Given input_tgt={args.input_tgt}, output_tgt={args.output_tgt}, "
            f"target_lang={args.target_lang}")
    args.input_src = args.input_src.expanduser()
    if args.input_tgt is not None:
        args.input_tgt = args.input_tgt.expanduser()
    args.output_src = args.output_src.expanduser()
    if args.output_tgt is not None:
        args.output_tgt = args.output_tgt.expanduser()
    return args


def main():
    args = get_args()
    count = 0
    if args.input_tgt is None:
        with open(args.input_src) as in_f, open(args.ouptut, 'w') as out_f:
            for l in in_f:
                l = l.strip()
                if detect(l) == args.input_lang:
                    count += 1
                    out_f.write(l + '\n')
    else:
        with open(args.input_src) as in_src, open(args.input_tgt) as in_tgt, open(args.output_src, 'w') as out_src, \
                open(args.output_tgt) as out_tgt:
            for src_l, tgt_l in zip(in_src, in_tgt):
                src_l = src_l.strip()
                tgt_l = tgt_l.strip()
                if detect(src_l) == args.source_lang and detect(tgt_l) == args.target_lang:
                    count += 1
                    out_src.write(src_l + '\n')
                    out_tgt.write(tgt_l + '\n')


if __name__ == "__main__":
    main()
