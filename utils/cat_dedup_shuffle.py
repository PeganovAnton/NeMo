import argparse
import random
from pathlib import Path


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
        pairs = []
        for s, t in zip(args.src, args.tgt):
            with s.open() as sf, t.open() as tf:
                for ss, ts in zip(sf, tf):
                    pairs.append((ss.strip(), ts.strip()))
        pairs = list(set(pairs))
        random.shuffle(pairs)
        with args.output_src.open() as sf, args.output_tgt.open() as tf:
            for p in pairs:
                sf.write(p[0] + '\n')
                tf.write(p[1] + '\n')
    else:
        sentences = []
        for s in args.src:
            with s.open() as sf:
                for ss in sf:
                    sentences.append((ss.strip()))
        sentences = list(set(sentences))
        random.shuffle(sentences)
        with args.output_src.open() as sf:
            for s in sentences:
                sf.write(s + '\n')


if __name__ == "__main__":
    main()
