 \
  && mkdir -p ${phase0_dir} \
  && if [ "$1" -eq "0" ]; then
    echo "********STARTING PHASE 0********" \
    && if [ ! -f ${phase0_dir}/best.ckpt ]; then
      if compgen -G ${phase0_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        resume0=true
      else
        resume0=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${phase0_train_data}/originals.txt \
          model.train_ds.tgt_file_name=${phase0_train_data}/translations.txt \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
          exp_manager.exp_dir=${phase0_dir} \
          trainer.max_epochs=100000 \
          +trainer.max_steps=${max_steps0} \
          +exp_manager.resume_if_exists=${resume0} \
          model.optim.lr=${phase0_lr}
    fi \
    && phase1_weights=${phase0_dir}/best.ckpt \
    && echo "********FINISHED PHASE 0********"
  else
    phase1_weights="$1"
  fi \
  && mkdir -p ${phase1_dir} \
  && if [ ! -f ${phase1_dir}/best.ckpt ]; then
    && if compgen -G ${phase1_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume1=true
    else
      resume1=false
    fi \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${phase_train_data1}/originals.txt \
        model.train_ds.tgt_file_name=${phase_train_data1}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${phase1_dir} \
        trainer.max_epochs=100000 \
        +trainer.max_steps=${max_steps1} \
        +exp_manager.resume_if_exists=${resume1} \
        +model.weights_checkpoint=${phase1_weights} \
        model.optim.lr=${back_translation_phases_lr}
  fi \
  && mkdir -p ${phase2_dir} \
  && if [ ! -f ${phase2_dir}/best.ckpt ]; then
    && if compgen -G ${phase2_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume2=true
    else
      resume2=false
    fi \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${phase2_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${phase2_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${phase2_dir} \
        +exp_manager.resume_if_exists=${resume2} \
        trainer.max_epochs=100000 \
        +trainer.max_steps=${max_steps2} \
        +model.weights_checkpoint=${phase1_dir}/best.ckpt \
        model.optim.lr=${back_translation_phases_lr}
  fi \
  && mkdir -p ${phase3_dir} \
  && if [ ! -f ${phase3_dir}/best.ckpt ]; then
    && if compgen -G ${phase3_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume3=true
    else
      resume3=false
    fi \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${phase3_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${phase3_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${phase3_dir} \
        trainer.max_epochs=100000 \
        +trainer.max_steps=${max_steps3} \
        +exp_manager.resume_if_exists=${resume3} \
        +model.weights_checkpoint=${phase2_dir}/best.ckpt \
        model.optim.lr=${back_translation_phases_lr}
  fi \
  && if [ ! -f ${phase4_dir}/best.ckpt ]; then
    && if compgen -G ${phase4_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
      resume4=true
    else
      resume4=false
    fi \
    && python3 train.py -cn ${base_conf} \
        trainer.num_nodes=${num_nodes} \
        trainer.gpus=${ngpus} \
        model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
        model.train_ds.src_file_name=${phase4_train_data}/originals.txt \
        model.train_ds.tgt_file_name=${phase4_train_data}/translations.txt \
        model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
        model.validation_ds.src_file_name=${valid_src} \
        model.validation_ds.tgt_file_name=${valid_ref} \
        model.test_ds.src_file_name=${test_src} \
        model.test_ds.tgt_file_name=${test_ref} \
        exp_manager.exp_dir=${phase4_dir} \
        trainer.max_epochs=100000 \
        +trainer.max_steps=${max_steps4} \
        +exp_manager.resume_if_exists=${resume4} \
        +model.weights_checkpoint=${phase3_dir}/best.ckpt \
        model.optim.lr=${back_translation_phases_lr}
  fi \
  && python3 test.py -cn ${base_conf} \
      trainer.num_nodes=${num_nodes} \
      trainer.gpus=${ngpus} \
      model.train_ds.src_file_name=${phase4_train_data}/originals.txt \
      model.train_ds.tgt_file_name=${phase4_train_data}/translations.txt \
      model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref} \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${phase4_dir}/best.ckpt