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
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Trainer
from hydra.utils import instantiate

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from omegaconf import DictConfig, OmegaConf


@dataclass
class MTEncDecConfig(NemoConfig):
    name: Optional[str] = 'MTEncDec'
    do_training: bool = True
    model: AAYNBaseConfig = AAYNBaseConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="en_de_8gpu")
def main(cfg: MTEncDecConfig) -> None:
    default_cfg = MTEncDecConfig()
    cfg = update_model_config(default_cfg, cfg)
    logging.info(f'Config: {cfg.pretty()}')
    trainer = Trainer(**cfg.trainer)
    if "exp_manager" in cfg and cfg.get("exp_manager") is not None:
        exp_manager(trainer, cfg.get("exp_manager", None))
    if "weights_checkpoint" in cfg.model and cfg.model.weights_checkpoint is not None:
        if cfg.model.weights_checkpoint.endswith(".ckpt"):
            transformer_mt = MTEncDecModel(cfg=cfg.model, trainer=trainer)
            transformer_mt.load_state_dict(torch.load(cfg.model.weights_checkpoint), strict=False)
        elif cfg.model.weights_checkpoint.endswith('.nemo'):
            with tarfile.open(cfg.model.weights_checkpoint, 'r:gz') as tar:
                names = tar.getnames()
                logging.info(f"Found names {names} in checkpoint {cfg.model.weights_checkpoint}")
                tokenizer_models = [n for n in names if n.endswith('.model')]
                if len(tokenizer_models) > 1:
                    raise ValueError(f"Found more than 1 tokenizer model: {tokenizer_models}")
                if len(tokenizer_models) == 0:
                    raise ValueError(f"Tokenizer model is not found. .nemo file contents are {names}")
                if "exp_manager" in cfg and cfg.get("exp_manager") is not None and "exp_dir" in cfg.exp_manager \
                        and cfg.exp_manager.exp_dir is not None:
                    working_dir = Path(cfg.exp_manager.exp_dir).resolve()
                else:
                    working_dir = Path.cwd()
                untarred_tokenizer_and_updated_config = working_dir / Path("tokenizer_dir")
                if is_global_rank_zero() and untarred_tokenizer_and_updated_config.exists():
                    shutil.rmtree(untarred_tokenizer_and_updated_config)
                    untarred_tokenizer_and_updated_config.mkdir()
                if is_global_rank_zero():
                    tar.extract(tokenizer_models[0], path=untarred_tokenizer_and_updated_config)
                nemo_tokenizer_model = untarred_tokenizer_and_updated_config / Path(tokenizer_models[0])
                if cfg.model.encoder_tokenizer.tokenizer_model is None:
                    logging.info(
                        f"There is an encoder tokenizer model specified in the config: "
                        f"{cfg.model.encoder_tokenizer.tokenizer_model}. Overwriting it with .nemo tokenizer model "
                        f"{nemo_tokenizer_model}")
                cfg.model.encoder_tokenizer.tokenizer_model = str(nemo_tokenizer_model)
                if cfg.model.decoder_tokenizer.tokenizer_model is None:
                    logging.info(
                        f"There is an decoder tokenizer model specified in the config: "
                        f"{cfg.model.decoder_tokenizer.tokenizer_model}. Overwriting it with .nemo tokenizer model "
                        f"{nemo_tokenizer_model}")
                cfg.model.decoder_tokenizer.tokenizer_model = str(nemo_tokenizer_model)
                config_path = untarred_tokenizer_and_updated_config / Path("updated_config.yaml")
                if is_global_rank_zero():
                    with config_path.open('w') as f:
                        f.write(OmegaConf.to_yaml(cfg.model))
                transformer_mt = MTEncDecModel.restore_from(
                    cfg.model.weights_checkpoint,
                    override_config_path=config_path,
                    map_location=torch.device('cpu'),
                    trainer=trainer
                )
                if transformer_mt._trainer is None:
                    logging.info(f"`_trainer attribute` of `transformer_mt` is expected to be set in `restore_from`. "
                                 f"Setting it manually.")
                    transformer_mt._trainer = trainer
                else:
                    logging.info(f"`_trainer` attribute of `transformer_mt` is set in `restore_from` as expected")
                transformer_mt.setup_training_data(cfg.model.train_ds)
                transformer_mt.setup_multiple_validation_data(None)
                transformer_mt.setup_multiple_test_data(None)
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
