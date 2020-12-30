stage0_dir=${result_dir}/stage0 \
  && mkdir -p ${stage0_dir} \
  && if [ "$1" -eq "0" ]; then
    echo "********STARTING PHASE 0********" \
    && if [ ! -f ${stage0_dir}/best.ckpt ]; then
      if compgen -G ${stage0_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        resume=true
      else
        resume=false
      fi \
      && stage_train_data="${data_path}/stage0" \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage_train_data}/originals.txt \
          model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
          exp_manager.exp_dir=${stage0_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=30000 \
          +exp_manager.resume_if_exists=${resume} \
          model.optim.lr=${stage0_lr}
    fi \
    && stage1_weights=${stage0_dir}/best.ckpt \
    && echo "********FINISHED PHASE 0********"
  else
    stage1_weights="$1"
  fi \
  && stage1_dir=${result_dir}/stage1 \
  && mkdir -p ${stage1_dir} \
  && if [ ! -f ${stage1_dir}/best.ckpt ]; then
    && if compgen -G ${stage1_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume=true
    else
      resume=false
    fi \
    && stage_train_data="${data_path}/stage1" \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${stage_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${stage1_dir} \
        trainer.max_epochs=${max_epochs} \
        +trainer.max_steps=50000 \
        +exp_manager.resume_if_exists=${resume} \
        +model.weights_checkpoint=${stage1_weights} \
        model.optim.lr=${back_translation_stages_lr}
  fi \
  && stage2_dir=${result_dir}/stage2 \
  && mkdir -p ${stage2_dir} \
  && if [ ! -f ${stage2_dir}/best.ckpt ]; then
    && if compgen -G ${stage2_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume=true
    else
      resume=false
    fi \
    && stage_train_data="${data_path}/stage2" \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${stage_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${stage2_dir} \
        +exp_manager.resume_if_exists=${resume} \
        trainer.max_epochs=${max_epochs} \
        +trainer.max_steps=75000 \
        +model.weights_checkpoint=${stage1_dir}/best.ckpt \
        model.optim.lr=${back_translation_stages_lr}
  fi \
  && stage3_dir=${result_dir}/stage3 \
  && mkdir -p ${stage3_dir} \
  && if [ ! -f ${stage3_dir}/best.ckpt ]; then
    && if compgen -G ${stage3_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume=true
    else
      resume=false
    fi \
    && stage_train_data="${data_path}/stage3" \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${stage_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${stage3_dir} \
        trainer.max_epochs=${max_epochs} \
        +trainer.max_steps=75000 \
        +exp_manager.resume_if_exists=${resume} \
        +model.weights_checkpoint=${stage2_dir}/best.ckpt \
        model.optim.lr=${back_translation_stages_lr}
  fi \
  && stage4_dir=${result_dir}/stage4 \
  && mkdir -p ${stage4_dir} \
  && if [ ! -f ${stage4_dir}/best.ckpt ]; then
    && if compgen -G ${stage4_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume=true
    else
      resume=false
    fi \
    && stage_train_data="${data_path}/stage4" \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${stage_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${stage4_dir} \
        trainer.max_epochs=${max_epochs} \
        +trainer.max_steps=100000 \
        +exp_manager.resume_if_exists=${resume} \
        +model.weights_checkpoint=${stage3_dir}/best.ckpt \
        model.optim.lr=${back_translation_stages_lr}
  fi \
  && python3 test.py -cn ${base_conf} \
      trainer.num_nodes=${num_nodes} \
      trainer.gpus=${ngpus} \
      model.train_ds.src_file_name=${stage_train_data}/originals.txt \
      model.train_ds.tgt_file_name=${stage_train_data}/translations.txt \
      model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref} \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${stage4_dir}/best.ckpt