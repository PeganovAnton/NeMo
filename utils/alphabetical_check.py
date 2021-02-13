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
    args = parser.parse_args()
    args.input_src = args.input_src.expanduser()
    args.output_src = args.output_src.expanduser()
    if args.input_target is not None:
        args.input_target = args.input_target.expanduser()
        args.output_target = args.output_target.expanduser()
        if args.tgt_lang is None:
            raise ValueError("If input target is provided than target language should be provided.")
    return args


def filter_singles(input, output, lang):
    alphabet = set(hg.Languages.get_alphabet([lang]))
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = out_dir / Path('.tmp')
    with tmp_file.open('w') as out_f, input.open() as in_f:
        for l in in_f:
            if set(l) & alphabet:
                out_f.write(l)
    tmp_file.rename(output)


def filter_pairs(input_src, input_tgt, output_src, output_tgt, src_lang, tgt_lang):
    src_alph = set(hg.Languages.get_alphabet([src_lang]))
    tgt_alph = set(hg.Languages.get_alphabet([tgt_lang]))
    out_src_dir = output_src.parent
    out_tgt_dir = output_tgt.parent
    out_src_dir.mkdir(parents=True, exist_ok=True)
    out_tgt_dir.mkdir(parents=True, exist_ok=True)
    tmp_src = out_src_dir / Path('.tmp_src')
    tmp_tgt = out_tgt_dir / Path('.tmp_tgt')
    with tmp_src.open('w') as out_s_f, tmp_tgt.open('w') as out_t_f, input_src.open() as in_s_f, \
            input_tgt.open('w') as in_t_f:
        for s_l, t_l in zip(in_s_f, in_t_f):
            if set(s_l) & src_alph and set(t_l) & tgt_alph:
                out_s_f.write(s_l)
                out_t_f.write(t_l)
    tmp_src.rename(output_src)
    tmp_tgt.rename(output_tgt)


def main():
    args = get_args()
    if args.input_target is None:
        filter_singles(args.input_src, args.output_src, args.src_lang)
    else:
        filter_pairs(args.input_src, args.input_tgt, args.output_src, args.output_tgt, args.src_lang, args.tgt_lang)


if __name__ == "__main__":
    main()
