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

import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Optional

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


@dataclass
class MTEncDecConfig(NemoConfig):
    model: AAYNBaseConfig = AAYNBaseConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="en_de_8gpu")
def main(cfg: MTEncDecConfig) -> None:
    if cfg.model.test_checkpoint_path is None:
        cfg.model.test_checkpoint_path = os.path.join(cfg.exp_manager.exp_dir, 'best.ckpt')
        if not os.path.exists(cfg.model.test_checkpoint_path):
            raise ValueError("Nor checkpoint for testing was provided, neither file best.ckpt is present in exp dir")
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    logging.info(f"Loading from checkpoint '{cfg.model.test_checkpoint_path}'")
    transformer_mt = MTEncDecModel.load_from_checkpoint(cfg.model.test_checkpoint_path)
    transformer_mt.setup_test_data(cfg.model.test_ds)
    trainer.test(transformer_mt)


if __name__ == '__main__':
    main()
