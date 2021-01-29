import argparse
import logging
import pickle
import shutil
import tarfile
import warnings
from pathlib import Path

import torch


logging.getLogger().setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to input .nemo file.",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Path to the output .nemo file.",
        type=Path,
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def fix_weights_dict(weights_dict):
    new_dict = dict()
    for key, value in weights_dict.items():
        if key.startswith('encoder.'):
            new_dict[key.replace('encoder.', 'encoder._encoder.')] = value
        elif key.startswith('decoder.'):
            new_dict[key.replace('decoder.', 'decoder._decoder.')] = value
        elif key.startswith('encoder_embedding'):
            new_dict[key.replace('encoder_embedding', 'encoder._embedding')] = value
        elif key.startswith('decoder_embedding'):
            new_dict[key.replace('decoder_embedding', 'decoder._embedding')] = value
        else:
            new_dict[key] = value
    return new_dict


def fix_nemo_checkpoint(args):
    tmp_dir = Path("tmp")
    with tarfile.open(args.input, "r:gz") as tar:
        init_names = [n for n in tar.getnames() if n != '.']
        tar.extractall(tmp_dir)
    model_names_file = None
    config_file_name = None
    tokenizer_file_name = None
    unknown_file_names = []
    logging.info(f"Content of archive {args.input}")
    for n in init_names:
        logging.info(f"{n}")
        if n.endswith('.ckpt'):
            if model_names_file is None:
                model_names_file = n
            else:
                raise ValueError(f"2 .ckpt files {model_names_file} and {n} are found in the archive {args.input}")
        elif n.endswith('.yaml'):
            if config_file_name is None:
                config_file_name = n
            else:
                raise ValueError(f"2 .yaml files {config_file_name} and {n} are found in the archive {args.input}")
        elif n.endswith('.model'):
            if tokenizer_file_name is None:
                tokenizer_file_name = n
            else:
                unknown_file_names.append(args.input)
                raise warnings.warn(f"More than 1 .model files are found in the archive {args.input}")
        else:
            warnings.warn(f"Unexpected file {n} is found in archive {args.input}.")
            unknown_file_names.append(n)
    try:
        weights_dict = fix_weights_dict(torch.load(tmp_dir / model_names_file))
    except pickle.UnpicklingError:
        raise ValueError(f".ckpt file in archive {args.input} is brocken. Cannot unpickle")
    torch.save(weights_dict, model_names_file)
    with tarfile.open(args.output, "w:gz") as tar:
        logging.info(f"Adding {model_names_file} to archive {args.output}")
        tar.add(tmp_dir / model_names_file, model_names_file)
        logging.info(f"Adding {config_file_name} to archive {args.output}")
        tar.add(tmp_dir / config_file_name, config_file_name)
        if tokenizer_file_name is not None:
            logging.info(f"Adding {tokenizer_file_name} to archive {args.output}")
            tar.add(tmp_dir / tokenizer_file_name, tokenizer_file_name)
        for n in unknown_file_names:
            logging.info(f"Adding {n} to archive {args.output}")
            tar.add(tmp_dir / n, n)
    shutil.rmtree(tmp_dir)


def fix_torch_checkpoint(args):
    try:
        weights_dict = fix_weights_dict(torch.load(args.input))
    except pickle.UnpicklingError:
        raise ValueError(f".ckpt file {args.input} is broken")
    torch.save(weights_dict, args.output)


def main():
    args = get_args()
    if args.input.suffix in ['.tgz', '.gz', '.nemo']:
        fix_nemo_checkpoint(args)
    else:
        fix_torch_checkpoint(args)


if __name__ == "__main__":
    main()
