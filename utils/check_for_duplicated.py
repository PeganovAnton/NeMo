import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file1",
        help="The first file with sentences"
    )
    parser.add_argument(
        "file2",
        help="The second file with sentences"
    )
    return parser.parse_args()



def main():
    args = get_args()
    with open(args.file1) as f1, open(args.file2) as f2:
        s1 = set(f1.readlines())
        s2 = set(f2.readlines())
    print("Number of duplicates:", len(s1 & s2))


if __name__ == "__main__":
    main()
