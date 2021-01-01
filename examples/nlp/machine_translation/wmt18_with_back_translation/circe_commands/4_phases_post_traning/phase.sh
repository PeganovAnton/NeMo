python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${nn} \
  trainer.gpus=${ng} \
  model.train_ds.tokens_in_batch=${bs} \
  model.train_ds.src_file_name=${td}/originals.txt \
  model.train_ds.tgt_file_name=${td}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  exp_manager.exp_dir=${phase0_dir} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps} \
  +exp_manager.resume_if_exists=${resume} \
  model.optim.lr=${lr}