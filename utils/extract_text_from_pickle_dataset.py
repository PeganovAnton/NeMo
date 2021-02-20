import argparse
import pickle
from pathlib import Path

from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input pickle file.",
        type=Path,
    )
    parser.add_argument(
        "--tokenizer-model",
        "-m",
        required=True,
        help="Path to the tokenizer model.",
        type=Path,
    )
    parser.add_argument(
        "--output-src",
        "-s",
        required=True,
        help="Path to the file where detokenized sources are saved.",
        type=Path,
    )
    parser.add_argument(
        "--output-tgt",
        "-t",
        required=True,
        help="Path to the file where detokenized targets are saved.",
        type=Path,
    )
    parser.add_argument(
        "--tokenizer-name",
        "-T",
        help="The name of the tokenizer available options are 'yttm' (default), 'sentencepiece', 'word', 'char' and "
             "some others. Full list is returned by `nemo.collections.nlp.modules.common.get_tokenizer_list`.",
        default='yttm',

    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.model = args.model.expanduser()
    args.output_src = args.output_src.expanduser()
    args.output_tgt = args.output_tgt.expanduser()
    return args


def main():
    args = get_args()
    with args.input.open('rb') as f:
        dataset = pickle.load(f)
    tokenizer = get_tokenizer(args.tokenizer_name, args.tokenizer_model, )
    with args.output_src.open('w') as out_s, args.output_tgt.open('w') as out_t:
        for b_i, b in dataset.batches.items():
            for s_tokens, t_tokens in zip(b['src'], b['tgt']):
                out_s.write(tokenizer.ids_to_text(s_tokens))
                out_s.write('\n')
                out_t.write(tokenizer.ids_to_text(t_tokens))
                out_t.write('\n')


if __name__ == "__main__":
    main()
