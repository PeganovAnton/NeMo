import argparse
import glob
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        help="Path to directory where folders rank0, rank1 and so on lay. "
             "Each of rank folders has to contain files originals.txt and "
             "translations.txt."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="A path to a directory where concatenated originals and translations "
             "will be put. The default is dir."
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.dir
    return args


def cat_rank_translations(input_dir, output_dir):
    rank_dirs = glob.glob(os.path.join(input_dir, "rank*"))
    with open(os.path.join(output_dir, "originals.txt"), 'w') as out_of, \
            open(os.path.join(output_dir, "translations.txt"), 'w') as out_tf:
        for rd in rank_dirs:
            with open(os.path.join(rd, "originals.txt")) as in_of, open(os.path.join(rd, "translations.txt")) as in_tf:
                original_lines = in_of.readlines()
                translation_lines = in_tf.readlines()
            n_lines = min(len(original_lines), len(translation_lines))
            out_of.write(''.join(original_lines[:n_lines]))
            out_tf.write(''.join(translation_lines[:n_lines]))


def main():
    args = get_args()
    cat_rank_translations(args.dir, args.output)


if __name__ == "__main__":
    main()
