pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && data_path=/workspace/mydatasets/sandeepsub/wmt18 \
  && bi_path=${data_path}/wmt18_en_de \
  && mono_en_path=${data_path}/wmt18_en_mono \
  && mono_de_path=${data_path}/wmt18_de_mono \
  && cat ${bi_path}/train.clean.* ${mono_en_path}/monolingual.2* ${mono_de_path}/monolingual.2* > all_text.txt \
  && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && python train.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.train_ds.tokens_in_batch=10000 \
      model.train_ds.src_file_name=${bi_path}/train.clean.de \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.en \
      model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
      exp_manager.exp_dir=/workspace/result \
  && python test.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.train_ds.src_file_name=${bi_path}/train.clean.de \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.en \
      model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
      exp_manager.exp_dir=/workspace/result