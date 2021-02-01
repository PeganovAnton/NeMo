import argparse
import warnings
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
        "--input-tgt",
        "-i",
        help="Path to the input target file.",
        type=Path,
    )
    parser.add_argument(
        "output_src",
        help="Path to the file which will contain output source.",
        type=Path,
    )
    parser.add_argument(
        "--output-tgt",
        "-o",
        help="Path to the output target file",
        type=Path,
    )
    parser.add_argument(
        "source_lang",
        help="Input language. See https://github.com/Mimino666/langdetect."
    )
    parser.add_argument(
        "--target-lang",
        "-L",
        help="Output language. See https://github.com/Mimino666/langdetect."
    )
    parser.add_argument(
        "removed_tgt",
        help="Path to file where removed target lines will be saved",
        type=Path,
    )
    parser.add_argument(
        "--removed-src",
        "-r",
        help="Path to file where removed source lines will be saved",
        type=Path,
    )
    args = parser.parse_args()
    if not (args.output_tgt is None and args.input_tgt is None and args.source_lang is None and args.removed_src is None \
            or args.output_tgt is not None and args.input_tgt is not None and args.target_lang is not None and args.removed_tgt is not None):
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
    args.removed_src = args.removed_src.expanduser()
    if args.removed_tgt is not None:
        args.removed_tgt = args.removed_tgt.expanduser()
    return args


def get_lang(line, fn, line_num):
    try:
       lang = detect(line)
    except:
       warnings.warn(f"No features found in line {repr(line)} with number {line_num} in file {fn}")
       lang = None
    return lang


def main():
    args = get_args()
    count = 0
    if args.input_tgt is None:
        with open(args.input_src) as in_f, open(args.ouptut, 'w') as out_f:
            for i, l in enumerate(in_f):
                l = l.strip()
                in_lang = get_lang(l, args.input_src, i)
                if in_lang is None:
                    continue
                if in_lang == args.input_lang:
                    count += 1
                    out_f.write(l + '\n')
    else:
        with open(args.input_src) as in_src, open(args.input_tgt) as in_tgt, open(args.output_src, 'w') as out_src, \
                open(args.output_tgt, 'w') as out_tgt:
            for i, (src_l, tgt_l) in enumerate(zip(in_src, in_tgt)):
                src_l = src_l.strip()
                tgt_l = tgt_l.strip()
                src_lang = get_lang(src_l, args.input_src, i)
                if src_lang is None:
                    continue
                tgt_lang = get_lang(tgt_l, args.input_tgt, i)
                if tgt_lang is None:
                    continue
                if src_lang == args.source_lang and tgt_lang == args.target_lang:
                    count += 1
                    out_src.write(src_l + '\n')
                    out_tgt.write(tgt_l + '\n')


if __name__ == "__main__":
    main()
