import argparse
from pathlib import Path


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