import argparse
import random
from collections import Counter
from itertools import zip_longest
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory",
        "-i",
        required=True,
        type=Path,
        help="Path to directory with files `originals.txt` and `translations.txt`."
    )
    parser.add_argument(
        "--reference-dataset-directory",
        "-r",
        required=True,
        type=Path,
        help="Path to reference dataset which is extended by sampled dataset. "
             "This is used for aligning of lengths distribution."
    )
    parser.add_argument(
        "--size",
        "-s",
        required=True,
        type=int,
        help="The number of pairs in the new dataset."
    )
    parser.add_argument(
        "--output-directory",
        "-o",
        required=True,
        type=Path,
        help="Path to directory where the new dataset will be saved. "
             "If this directory does not exist it will be created."
    )
    parser.add_argument(
        "--bucket-min-size",
        "-b",
        type=int,
        help="Minimum number of sentences in from which sampling is done."
    )
    args = parser.parse_args()
    args.input_directory = args.input_directory.expanduser()
    args.output_directory = args.output_directory.expanduser()
    args.reference_dataset_directory = args.reference_dataset_directory.expanduser()
    return args


def read_dataset(originals_file, translations_file):
    pairs = []
    with originals_file.open() as of, translations_file.open() as tf:
        for o, t in zip_longest(of, tf):
            if o is None or t is None:
                raise ValueError(
                    "The number of lines in translations file is different from the number of lines in originals file.")
            pairs.append((o.strip(), t.strip()))
    return sorted(pairs, key=lambda x: len(x[0]) + len(x[1]))


def write_dataset(pairs, path_to_dir):
    path_to_dir = path_to_dir.expanduser()
    path_to_dir.mkdir(parents=True, exist_ok=True)
    originals_path = path_to_dir / Path("originals.txt")
    translations_path = path_to_dir / Path("translations.txt")
    with originals_path.open('w') as of, translations_path.open('w') as tf:
        for o, t in pairs:
            of.write(o + '\n')
            tf.write(t + '\n')


def collect_dataset_len_stats(originals_file, translations_file):
    c = Counter()
    with originals_file.open() as of, translations_file.open() as tf:
        for o, t in zip_longest(of, tf):
            if o is None or t is None:
                raise ValueError(
                    "The number of lines in translations file is different from the number of lines in originals file.")
            c.update([len(o.strip()) + len(t.strip())])
    return dict(sorted(c.items(), key=lambda x: x[0]))


def sample_cleverly(pairs, reference_len_counts, n, bucket_min_size):
    if not pairs:
        raise ValueError("`pairs` is empty. Nothing to sample from.")
    reference_dataset_size = sum(reference_len_counts.values())
    if reference_dataset_size == 0:
        raise ValueError("Reference dataset is empty.")
    buckets = []
    ref_bucket_sizes = []
    bucket_new_count_to_ref_count_fractions = []
    curr_pair_len = 0
    curr_bucket = []
    curr_ref_bucket_size = 0
    ref_counts_iter = iter(reference_len_counts.items())
    try:
        ref_pair_len, ref_count = next(ref_counts_iter)
    except StopIteration:
        raise ValueError("Reference dataset is empty.")
    for i, p in enumerate(pairs):
        new_pair_len = len(p[0]) + len(p[1])
        if new_pair_len != curr_pair_len \
                and len(curr_bucket) >= bucket_min_size \
                and curr_ref_bucket_size >= bucket_min_size:
            if new_pair_len < curr_pair_len:
                raise ValueError("Pairs have to be sorted in a non decreasing order.")
            ref_bucket_sizes.append(curr_ref_bucket_size)
            bucket_new_count_to_ref_count_fractions.append(
                len(curr_bucket) / curr_ref_bucket_size if curr_ref_bucket_size > 0 else float('+inf')
            )
            buckets.append(curr_bucket)
            curr_ref_bucket_size = 0
            while ref_pair_len <= new_pair_len:
                curr_ref_bucket_size += ref_count
                try:
                    ref_pair_len, ref_count = next(ref_counts_iter)
                except StopIteration:
                    ref_count = 0
                    break
            curr_bucket = [i]
            curr_pair_len = new_pair_len
        else:
            curr_pair_len = new_pair_len
            while ref_pair_len <= new_pair_len:
                curr_ref_bucket_size += ref_count
                try:
                    ref_pair_len, ref_count = next(ref_counts_iter)
                except StopIteration:
                    ref_count = 0
                    break

            curr_bucket.append(i)
    buckets, ref_bucket_sizes, bucket_new_count_to_ref_count_fractions = zip(
        *sorted(zip(buckets, ref_bucket_sizes, bucket_new_count_to_ref_count_fractions), key=lambda x: x[2]))
    sampled_indices = []
    remain_to_sample = n
    remain_in_reference_dataset = reference_dataset_size
    for i, (ref_bucket_size, bucket) in enumerate(zip(ref_bucket_sizes, buckets)):
        if len(bucket) / ref_bucket_size < remain_to_sample / remain_in_reference_dataset:
            sampled_indices += bucket
            remain_to_sample -= len(bucket)
        else:
            if i < len(ref_bucket_sizes) - 1:
                bucket_sample = random.sample(bucket, round(ref_bucket_size * remain_to_sample / remain_in_reference_dataset))
            else:
                assert len(bucket) >= remain_to_sample
                bucket_sample = random.sample(bucket, remain_to_sample)
            sampled_indices += bucket_sample
            remain_to_sample -= len(bucket_sample)
        remain_in_reference_dataset -= ref_bucket_size
    sampled_pairs = [pairs[i] for i in sampled_indices]
    return sampled_pairs


def get_dataset_files(dir_):
    return dir_.expanduser() / Path("originals.txt"), dir_.expanduser() / Path("translations.txt")


def main():
    args = get_args()
    input_pairs = read_dataset(*get_dataset_files(args.input_directory))
    reference_counts = collect_dataset_len_stats(*get_dataset_files(args.reference_dataset_directory))
    output_pairs = sample_cleverly(input_pairs, reference_counts, args.size, args.bucket_min_size)
    write_dataset(output_pairs, args.output_directory)


if __name__ == '__main__':
    main()
