export stage1_dir=${result_dir}/stage1 \
  && mkdir -p ${stage1_dir} \
  && if [ "$1" -eq "0" ]; then
      export stage0_dir=${result_dir}/stage0 \
      && mkdir -p ${stage0_dir} \
      && if [ ! -f ${stage0_dir}/best.ckpt ]; then
          cat ${par_train_src} >"${stage0_dir}/originals.txt" \
          && cat ${par_train_ref} >"${stage0_dir}/translations.txt" \
          && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage0_dir} \
          && if compgen -G ${stage0_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
            export resume=true
          else
            export resume=false
          fi \
          && python3 train.py -cn ${base_conf} \
              trainer.num_nodes=${num_nodes} \
              trainer.gpus=${ngpus} \
              model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
              model.train_ds.src_file_name=${stage0_dir}/originals.txt \
              model.train_ds.tgt_file_name=${stage0_dir}/translations.txt \
              model.validation_ds.src_file_name=${valid_src} \
              model.validation_ds.tgt_file_name=${valid_ref} \
              model.test_ds.src_file_name=${test_src} \
              model.test_ds.tgt_file_name=${test_ref} \
              model.tokenizer.tokenizer_model=${tok_model} \
              exp_manager.exp_dir=${stage0_dir} \
              trainer.max_epochs=${max_epochs} \
              +trainer.max_steps=30000 \
              +exp_manager.resume_if_exists=${resume}
      fi \
      && export stage1_weights=${stage0_dir}/best.ckpt
    else
      export stage1_weights="$1"
    fi \
  && export stage1_dir=${result_dir}/stage1 \
  && mkdir -p ${stage1_dir} \
  && if [ ! -f ${stage1_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/3M/translations.txt >"${stage1_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/3M/originals.txt >"${stage1_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage1_dir} \
      && if compgen -G ${stage1_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage1_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage1_dir}/translations.txt \
          model.tokenizer.tokenizer_model=${tok_model} \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage1_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=50000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage1_weights}
    fi \
  && export stage2_dir=${result_dir}/stage2 \
  && mkdir -p ${stage2_dir} \
  && if [ ! -f ${stage2_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/6M/translations.txt >"${stage2_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/6M/originals.txt >"${stage2_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage2_dir} \
      && if compgen -G ${stage2_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage2_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage2_dir}/translations.txt \
          model.tokenizer.tokenizer_model=${tok_model} \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage2_dir} \
          +exp_manager.resume_if_exists=${resume} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=75000 \
          +model.weights_checkpoint=${stage1_dir}/best.ckpt
     fi \
  && export stage3_dir=${result_dir}/stage3 \
  && mkdir -p ${stage3_dir} \
  && if [ ! -f ${stage3_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/12M/translations.txt >"${stage3_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/12M/originals.txt >"${stage3_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage3_dir} \
      && if compgen -G ${stage3_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage3_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage3_dir}/translations.txt \
          model.tokenizer.tokenizer_model=${tok_model} \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage3_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=75000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage2_dir}/best.ckpt
     fi \
  && export stage4_dir=${result_dir}/stage4 \
  && mkdir -p ${stage4_dir} \
  && if [ ! -f ${stage4_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/24M/translations.txt >"${stage4_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/24M/originals.txt >"${stage4_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage4_dir} \
      && if compgen -G ${stage4_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage4_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage4_dir}/translations.txt \
          model.tokenizer.tokenizer_model=${tok_model} \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage4_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=100000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage3_dir}/best.ckpt
     fi \
  && python3 test.py -cn ${base_conf} \
      trainer.num_nodes=${num_nodes} \
      trainer.gpus=${ngpus} \
      model.train_ds.src_file_name=${stage4_dir}/originals.txt \
      model.train_ds.tgt_file_name=${stage4_dir}/translations.txt \
      model.tokenizer.tokenizer_model=${tok_model} \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref} \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${stage4_dir}/best.ckpt