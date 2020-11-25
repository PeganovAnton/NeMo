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
from omegaconf import DictConfig

from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero


@hydra_runner(config_path="conf", config_name="en_de_8gpu")
def main(cfg: DictConfig) -> None:
    cfg.exp_manager.name = None
    cfg.exp_manager.exp_dir = None
    cfg.exp_manager.create_tensorboard_logger = False
    cfg.exp_manager.create_checkpoint_callback = False
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(resume_from_checkpoint=cfg.model.train_checkpoint_path)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    transformer_mt = TransformerMTModel(cfg.model, trainer=trainer)
    trainer.fit(transformer_mt)
    if is_global_rank_zero():
        best_ckpt_path = os.path.join(cfg.exp_manager.exp_dir, 'best.ckpt')
        if os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
        os.symlink(trainer.checkpoint_callback.best_model_path, best_ckpt_path)


if __name__ == '__main__':
    main()
