pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && pip install webdataset \
  && pip install transformers==3.5.0 \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export mono_data_path=/workspace/mydatasets/apeganov/wmt18 \
  && export bi_path=/workspace/mydatasets/apeganov/wmt/wmt18_en_de/parallel \
  && export mono_en_path=${mono_data_path}/wmt18_en_mono \
  && export mono_de_path=${mono_data_path}/wmt18_de_mono \
  && cat ${bi_path}/train.clean.en ${bi_path}/train.clean.de ${mono_en_path}/monolingual.25000000 ${mono_de_path}/monolingual.25000000 > all_text.txt \
  && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && python train.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.train_ds.tokens_in_batch=6000 \
      model.train_ds.src_file_name=${bi_path}/train.clean.de \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.en \
      model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
      exp_manager.exp_dir=/workspace/result \
      trainer.max_epochs=50 \
  && python test.py -cn wmt16/de_en_8gpu \
      trainer.gpus=16 \
      model.train_ds.src_file_name=${bi_path}/train.clean.de \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.en \
      model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
      exp_manager.exp_dir=/workspace/result