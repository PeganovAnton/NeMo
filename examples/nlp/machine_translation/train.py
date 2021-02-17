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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import instantiate

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class MTEncDecConfig(NemoConfig):
    model: AAYNBaseConfig = AAYNBaseConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="en_de_8gpu")
def main(cfg: MTEncDecConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = instantiate(cfg.trainer)
    if "exp_manager" in cfg and cfg.get("exp_manager") is not None:
        exp_manager(trainer, cfg.get("exp_manager", None))
    if "weights_checkpoint" in cfg.model and cfg.model.weights_checkpoint is not None:
        transformer_mt = MTEncDecModel(cfg=cfg.model, trainer=trainer)
        print("transformer_mt.beam_search:", transformer_mt.beam_search)
        transformer_mt.load_state_dict(torch.load(cfg.model.weights_checkpoint), strict=False)
        #transformer_mt = MTEncDecModel.load_from_checkpoint(cfg.model.weights_checkpoint, cfg=cfg.model, trainer=trainer)
        #transformer_mt._trainer = trainer
        # transformer_mt.setup_training_data(cfg.model.train_ds)
        # transformer_mt.setup_validation_data(cfg.model.validation_ds)
        # transformer_mt.setup_test_data(cfg.model.test_ds)
    else:
        transformer_mt = MTEncDecModel(cfg.model, trainer=trainer)
    trainer.fit(transformer_mt)
    if is_global_rank_zero():
        if "exp_manager" not in cfg \
                or cfg.exp_manager is None \
                or "exp_dir" not in cfg.exp_manager \
                or cfg.exp_manager.exp_dir is None:
            best_ckpt_path = os.path.join(
                str(Path(trainer.checkpoint_callback.best_model_path).parents[3]),
                'best.ckpt'
            )
        else:
            best_ckpt_path = os.path.join(cfg.exp_manager.exp_dir, 'best.ckpt')
        print(f"saving link to the best checkpoint into '{best_ckpt_path}'")
        if os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
        os.symlink(trainer.checkpoint_callback.best_model_path, best_ckpt_path)


if __name__ == '__main__':
    main()
