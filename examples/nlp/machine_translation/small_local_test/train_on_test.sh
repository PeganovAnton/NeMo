data_path=~/data/wmt14_en_de_train_on_test
tok_model="${data_path}/bpe_32k_en_de_yttm.model"
python enc_dec_nmt.py \
  --config-path=conf \
  --config-name=aayn_small \
  trainer.gpus=[0,1] \
  ~trainer.max_epochs \
  +trainer.max_steps=16000 \
  model.beam_size=4 \
  model.max_generation_delta=5 \
  model.label_smoothing=0.1 \
  model.encoder_tokenizer.tokenizer_model="${tok_model}" \
  model.encoder_tokenizer.bpe_dropout=0.1 \
  model.decoder_tokenizer.tokenizer_model="${tok_model}" \
  model.decoder_tokenizer.bpe_dropout=0.1 \
  model.train_ds.src_file_name="${data_path}/train.en" \
  model.train_ds.tgt_file_name="${data_path}/train.de" \
  model.validation_ds.src_file_name="${data_path}/valid.en" \
  model.validation_ds.tgt_file_name="${data_path}/valid.de" \
  model.test_ds.src_file_name="${data_path}/test.en" \
  model.test_ds.tgt_file_name="${data_path}/test.de" \
  model.optim.lr=0.0003  \
  model.optim.sched.warmup_ratio=0.1 \
  exp_manager.wandb_logger_kwargs.name=1st_trial \
  exp_manager.wandb_logger_kwargs.project=nmt_aayn_en_de_small_train_on_test_v0 \
