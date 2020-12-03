# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.machine_translation import TranslationOneSideDataset
from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--text2translate", type=str, required=True, help="")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="")
    parser.add_argument("--max_num_tokens_in_batch", type=int, required=True, help="")
    parser.add_argument("--translations", type=str, required=True, help="")
    parser.add_argument("--originals", type=str, required=True, help="")
    args = parser.parse_args()
    transformer_mt = TransformerMTModel.load_from_checkpoint(args.model)
    transformer_mt.teacher_forcing_forward = False
    transformer_mt.pad_beam_search_results_to_max_seq_len = True
    device = torch.device('cuda:0')
    transformer_mt.to(device)
    parallel_model = DataParallel(transformer_mt)
    src_tokenizer = get_tokenizer(tokenizer_name='yttm', tokenizer_model=args.tokenizer_model)
    tgt_tokenizer = src_tokenizer
    dataset = TranslationOneSideDataset(
        src_tokenizer,
        args.text2translate,
        tokens_in_batch=args.max_num_tokens_in_batch,
        max_seq_length=2048,
        cache_ids=True,
    )
    loader = DataLoader(dataset, batch_size=1, pin_memory=False)
    num_translated_sentences = 0
    parallel_model.eval()
    with open(args.translations, 'w') as tf, open(args.originals, 'w') as of:
        for batch_idx, batch in enumerate(loader):
            for i in range(len(batch)):
                if batch[i].ndim == 3:
                    batch[i] = batch[i].squeeze(dim=0)
                batch[i] = batch[i].to(device)
            src_ids, src_mask, sent_ids = batch
            if batch_idx % 100 == 0:
                logging.info(f"{batch_idx} batches and {num_translated_sentences} sentences were translated")
            num_translated_sentences += len(src_ids)
            _, translations = parallel_model(src_ids, src_mask)
            translations = translations.cpu().numpy()
            for t in translations:
                tf.write(tgt_tokenizer.ids_to_text(t) + '\n')
            for o in src_ids:
                of.write(src_tokenizer.ids_to_text(o) + '\n')
    logging.info("done")


if __name__ == '__main__':
    main()
