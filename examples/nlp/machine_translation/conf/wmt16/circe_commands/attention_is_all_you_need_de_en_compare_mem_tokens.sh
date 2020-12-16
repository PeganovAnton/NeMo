export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export data_path=/workspace/mydatasets/apeganov/wmt18 \
  && export bi_path=${data_path}/parallel_clean \
  && export valid_src=${bi_path}/newstest2013-de-en.src \
  && export valid_ref=${bi_path}/newstest2013-de-en.ref \
  && export test_src=${bi_path}/newstest2014-de-en.src \
  && export test_ref=${bi_path}/newstest2014-de-en.ref \
  && export par_train_src=${bi_path}/train.clean.filter.de.shuffled \
  && export par_train_ref=${bi_path}/train.clean.filter.en.shuffled \
  && export tok_model=${data_path}/bpe_35k_en_de_yttm.model \
  && python3 train.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.train_ds.tokens_in_batch=10000 \
      model.tokenizer.tokenizer_model=${tok_model} \
      model.train_ds.src_file_name=${par_train_src} \
      model.train_ds.tgt_file_name=${par_train_ref} \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref}\
      exp_manager.exp_dir=/workspace/mem_tokens_2nd_trial_compare \
      trainer.max_epochs=50 \
  && python3 test.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.tokenizer.tokenizer_model=${tok_model} \
      model.train_ds.src_file_name=${par_train_src} \
      model.train_ds.tgt_file_name=${par_train_ref} \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref} \
      exp_manager.exp_dir=/workspace/mem_tokens_2nd_trial_compare