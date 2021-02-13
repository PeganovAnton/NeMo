import argparse
from pathlib import Path

import homoglyphs as hg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-src",
        "-s",
        required=True,
        type=Path,
        help="Path to the input source file."
    )
    parser.add_argument(
        "--input-tgt",
        "-t",
        type=Path,
        help="Path to the input target file."
    )
    parser.add_argument(
        "--output-src",
        "-S",
        required=True,
        type=Path,
        help="Path to the output source file."
    )
    parser.add_argument(
        "--output-tgt",
        "-T",
        type=Path,
        help="Path to the output target file."
    )
    parser.add_argument(
        "--removed_src",
        "-r",
        required=True,
        type=Path,
        help="Path to a file where removed source lines will be put."
    )
    parser.add_argument(
        "--removed_tgt",
        "-R",
        type=Path,
        help="Path to a file where removed target lines will be put."
    )
    parser.add_argument(
        "--src_lang",
        "-l",
        required=True,
        help="The source language. Possible options are listed here https://github.com/life4/homoglyphs or with "
             "`homoglyphs.Languages.get_all()`."
    )
    parser.add_argument(
        "--tgt_lang",
        "-L",
        help="The target language. Possible options are listed here https://github.com/life4/homoglyphs or with "
             "`homoglyphs.Languages.get_all()`."
    )
    parser.add_argument(
        "--fraction",
        "-f",
        help="The fraction of characters which have to belong to the specified language. By default, there have to be "
             "at least 1 character belonging to the specified language.",
        type=float,
    )
    args = parser.parse_args()
    args.input_src = args.input_src.expanduser()
    args.output_src = args.output_src.expanduser()
    args.removed_src = args.removed_src.expanduser()
    if args.input_tgt is not None:
        args.input_tgt = args.input_tgt.expanduser()
        args.output_tgt = args.output_tgt.expanduser()
        args.removed_tgt = args.removed_tgt.expanduser()
        if args.tgt_lang is None:
            raise ValueError("If input target is provided than target language should be provided.")
    return args


def filter_singles(input, output, removed, lang, fraction):
    alphabet = set(hg.Languages.get_alphabet([lang]))
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = out_dir / Path('.tmp')
    with tmp_file.open('w') as out_f, input.open() as in_f, removed.open('w') as r_f:
        for l in in_f:
            intersection = set(l) & alphabet
            if fraction is None and intersection or len(intersection) / len(l) > fraction:
                out_f.write(l)
            else:
                r_f.write(l)
    tmp_file.rename(output)


def filter_pairs(input_src, input_tgt, output_src, output_tgt, removed_src, removed_tgt, src_lang, tgt_lang, fraction):
    src_alph = set(hg.Languages.get_alphabet([src_lang]))
    tgt_alph = set(hg.Languages.get_alphabet([tgt_lang]))
    out_src_dir = output_src.parent
    out_tgt_dir = output_tgt.parent
    out_src_dir.mkdir(parents=True, exist_ok=True)
    out_tgt_dir.mkdir(parents=True, exist_ok=True)
    tmp_src = out_src_dir / Path('.tmp_src')
    tmp_tgt = out_tgt_dir / Path('.tmp_tgt')
    with tmp_src.open('w') as out_s_f, tmp_tgt.open('w') as out_t_f, input_src.open() as in_s_f, \
            input_tgt.open() as in_t_f, removed_src.open('w') as r_s_f, removed_tgt.open('w') as r_t_f:
        for s_l, t_l in zip(in_s_f, in_t_f):
            s_int = set(s_l) & src_alph
            t_int = set(t_l) & tgt_alph
            if fraction is None and s_int and t_int or \
                    len(s_int) / len(s_l) > fraction and len(t_int) / len(t_l) > fraction:
                out_s_f.write(s_l)
                out_t_f.write(t_l)
            else:
                r_s_f.write(s_l)
                r_t_f.write(t_l)
    tmp_src.rename(output_src)
    tmp_tgt.rename(output_tgt)


def main():
    args = get_args()
    if args.input_tgt is None:
        filter_singles(args.input_src, args.output_src, args.removed_src, args.src_lang, args.fraction)
    else:
        filter_pairs(
            args.input_src,
            args.input_tgt,
            args.output_src,
            args.output_tgt,
            args.removed_src,
            args.removed_tgt,
            args.src_lang,
            args.tgt_lang,
            args.fraction,
        )


if __name__ == "__main__":
    main()
