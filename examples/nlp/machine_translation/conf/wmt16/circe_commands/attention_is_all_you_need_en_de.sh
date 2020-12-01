pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && pip install webdataset \
  && pip install transformers=3.5.0 \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && mono_data_path=/workspace/mydatasets/sandeepsub/wmt18 \
  && bi_path=/workspace/mydatasets/sandeepsub/wmt/wmt18_en_de/parallel \
  && mono_en_path=${mono_data_path}/wmt18_en_mono \
  && mono_de_path=${mono_data_path}/wmt18_de_mono \
  && cat ${bi_path}/train.clean.* ${mono_en_path}/monolingual.2* ${mono_de_path}/monolingual.2* > all_text.txt \
  && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && python train.py -cn wmt16/en_de_8gpu \
      trainer.gpus=16 \
      model.train_ds.tokens_in_batch=6000 \
      model.train_ds.src_file_name=${bi_path}/train.clean.en \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.de \
      model.validation_ds.src_file_name=${bi_path}/wmt13-en-de.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-en-de.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-en-de.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-en-de.ref \
      exp_manager.exp_dir=/workspace/result \
      trainer.max_epochs=50 \
  && python test.py -cn wmt16/en_de_8gpu \
      trainer.gpus=16 \
      model.train_ds.src_file_name=${bi_path}/train.clean.en \
      model.train_ds.tgt_file_name=${bi_path}/train.clean.de \
      model.validation_ds.src_file_name=${bi_path}/wmt13-en-de.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-en-de.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-en-de.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-en-de.ref \
      exp_manager.exp_dir=/workspace/result