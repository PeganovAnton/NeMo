import argparse
import shutil
import tarfile
import warnings
from pathlib import Path

import torch


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
        init_names = tar.getnames()
        tar.extractall(tmp_dir)
    model_names_file = None
    config_file_name = None
    tokenizer_file_name = None
    unknown_file_names = []
    for n in init_names:
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
    weights_dict = fix_weights_dict(torch.load(model_names_file))
    torch.save(weights_dict, model_names_file)
    with tarfile.open(args.output, "w:gz") as tar:
        tar.add(model_names_file)
        tar.add(config_file_name)
        if tokenizer_file_name is not None:
            tar.add(tokenizer_file_name)
        for n in unknown_file_names:
            tar.add(n)
    shutil.rmtree(tmp_dir)


def fix_torch_checkpoint(args):
    weights_dict = fix_weights_dict(torch.load(args.input))
    torch.save(weights_dict, args.output)


def main():
    args = get_args()
    if args.input.endswith(".nemo") or args.input.endswith(".ckpt"):
        fix_nemo_checkpoint(args)
    else:
        fix_torch_checkpoint(args)


if __name__ == "__main__":
    main()
