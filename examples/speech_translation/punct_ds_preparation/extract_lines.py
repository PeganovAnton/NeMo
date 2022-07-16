import argparse
import os
import random
from pathlib import Path
from subprocess import PIPE, run
from typing import Union

from tqdm import tqdm


BUFFER_SIZE = 2 ** 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--extracted_file", type=Path, required=True)
    parser.add_argument("--remaining_file", type=Path, required=True)
    parser.add_argument(
        "--num_lines_to_extract",
        type=int,
        default=5000,
    )
    for name in ["input_file", "extracted_file", "remaining_name"]:
        setattr(parser, name, getattr(parser, name).expanduser())
    args = parser.parse_args()
    return args


def get_num_lines(input_file: Union[str, os.PathLike]) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def main() -> None:
    args = parse_args()
    num_lines = get_num_lines(args.input_file)
    extracted_line_indices = set([random.randrange(0, num_lines) for _ in range(args.num_lines_to_extract)])
    with args.input_file.open(buffering=BUFFER_SIZE) as in_f, \
            args.extracted_file.open('w', buffering=BUFFER_SIZE) as e_f, \
            args.remaining_file.open('w', buffering=BUFFER_SIZE) as r_f:
        for i, line in enumerate(tqdm(in_f, unit="line", total=num_lines, desc="Extracting lines")):
            if i in extracted_line_indices:
                e_f.write(line)
            else:
                r_f.write(line)


if __name__ == "__main__":
    main()