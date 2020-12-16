export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export base_conf=wmt16/de_en_8gpu \
  && export num_nodes="$1" \
  && export ngpus="$2" \
  && export train_n_tokens_in_batch=11000 \
  && export max_epochs=1000 \
  && export data_path=/workspace/mydatasets/apeganov/wmt18 \
  && export bi_path=/workspace/mydatasets/apeganov/wmt18/parallel_clean \
  && export par_train_src=${bi_path}/train.clean.filter.en.shuffled \
  && export par_train_ref=${bi_path}/train.clean.filter.de.shuffled \
  && export bt_path=/workspace/mydatasets/apeganov/wmt18/en_de_with_backstranslation_not_full_6.12.2020 \
  && export mono_data_path=${data_path}/de_mono_partially_translated_06.12.20_sampled \
  && export result_dir=/workspace/result_en_de_back_translation_4_stages_1st_trial \
  && export valid_src=${bi_path}/newstest2013-en-de.src \
  && export valid_ref=${bi_path}/newstest2013-en-de.ref \
  && export test_src=${bi_path}/newstest2014-en-de.src \
  && export test_ref=${bi_path}/newstest2014-en-de.ref \
  && export par_train_src=${bi_path}/train.clean.filter.en.shuffled \
  && export par_train_ref=${bi_path}/train.clean.filter.de.shuffled \
  && export tok_model=${data_path}/bpe_35k_en_de_yttm.model