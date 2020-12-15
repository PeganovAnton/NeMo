pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && pip install webdataset \
  && pip install transformers==3.5.0 \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export base_conf=de_en_debug \
  && export ngpus=2 \
  && export train_n_tokens_in_batch=512 \
  && export max_epochs=1000 \
  && export result_dir=nemo_experiments/debug_back_translation \
  && export stage1_dir=${result_dir}/stage1 \
  && if [ ! -f ${stage1_dir}/best.ckpt ]; then
      if compgen -G ${stage1_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          exp_manager.exp_dir=${stage1_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=3 \
          +exp_manager.resume_if_exists=${resume}
     fi \
  && export stage2_dir=${result_dir}/stage2 \
  && if [ ! -f ${stage2_dir}/best.ckpt ]; then
      if compgen -G ${stage2_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          exp_manager.exp_dir=${stage2_dir} \
          +exp_manager.resume_if_exists=${resume} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=4 \
          +model.weights_checkpoint=${stage1_dir}/best.ckpt
     fi \
  && export stage3_dir=${result_dir}/stage3 \
  && if [ ! -f ${stage3_dir}/best.ckpt ]; then
      if compgen -G ${stage3_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          exp_manager.exp_dir=${stage3_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=5 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage2_dir}/best.ckpt
     fi \
  && export stage4_dir=${result_dir}/stage4 \
  && if [ ! -f ${stage4_dir}/best.ckpt ]; then
      if compgen -G ${stage4_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          exp_manager.exp_dir=${stage4_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=6 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage3_dir}/best.ckpt
     fi \
  && python3 test.py -cn ${base_conf} \
      trainer.gpus=${ngpus} \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${stage4_dir}/best.ckpt
