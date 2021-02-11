import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        "-s",
        nargs="+",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--tgt",
        "-t",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--output_src",
        "-o",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output_tgt",
        "-r",
        type=Path
    )
    args = parser.parse_args()
    args.src = [p.expanduser() for p in args.src]
    if args.tgt is not None:
        args.tgt = [p.expanduser() for p in args.tgt]
    args.output_src = args.output_src.expanduser()
    if args.output_tgt is not None:
        args.output_tgt = args.output_tgt.expanduser()
    return args


def main():
    args = get_args()
    random.seed(42)
    if args.tgt is not None:
        pairs = set()
        for s, t in zip(args.src, args.tgt):
            with s.open() as sf, t.open() as tf:
                for ss, ts in zip(sf, tf):
                    pairs.add((ss.strip(), ts.strip()))
        pairs = list(pairs)
        random.shuffle(pairs)
        args.output_src.parent.mkdir(parents=True, exist_ok=True)
        args.output_tgt.parent.mkdir(parents=True, exist_ok=True)
        with args.output_src.open('w') as sf, args.output_tgt.open('w') as tf:
            for p in pairs:
                sf.write(p[0] + '\n')
                tf.write(p[1] + '\n')
    else:
        data = np.concatenate([np.loadtxt(fn, dtype=str, delimiter='\t', encoding='utf-8') for fn in args.src], axis=0)
        print("Number of sentences before deduplication:", data.shape[0])
        unique = pd.unique(data)
        print("Number of sentences after deduplication:", unique.shape[0])
        np.random.shuffle(unique)
        args.output_src.parent.mkdir(parents=True, exist_ok=True)
        with args.output_src.open('w') as sf:
            for s in unique:
                sf.write(s)


if __name__ == "__main__":
    main()
