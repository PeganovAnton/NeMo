pip3 install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && pip install webdataset \
  && pip install transformers==3.5.0 \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export python3PATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export base_conf=wmt16/de_en_8gpu \
  && export ngpus=16 \
  && export train_n_tokens_in_batch=6000 \
  && export max_epochs=1000 \
  && export mono_data_path=/workspace/mydatasets/apeganov/wmt18 \
  && export bi_path=/workspace/mydatasets/apeganov/wmt/wmt18_en_de/parallel \
  && export bt_path=/workspace/mydatasets/apeganov/wmt18/en_de_with_backstranslation_not_full_6.12.2020 \
  && export mono_en_path=${mono_data_path}/wmt18_en_mono \
  && export mono_de_path=${mono_data_path}/wmt18_de_mono \
  && cat ${bt_path}/src.en ${bt_path}/tgt.de ${bt_path}/src.de ${bt_path}/tgt.en > all_text.txt \
  && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && export result_dir=/workspace/result_de_en_back_translation_4_stages_1st_trial \
  && export stage1_dir=${result_dir}/stage1 \
  && if [ ! -f ${stage1_dir}/best.ckpt ]; then
      cat ${bi_path}/train.clean.de ${mono_data_path}/3M_en_mono_partiaally_translated_06.12.20/translations.txt > ${stage1_dir}/src.de \
      && cat ${bi_path}/train.clean.en ${mono_data_path}/3M_en_mono_partiaally_translated_06.12.20/originals.txt > ${stage1_dir}/tgt.en \
      && if compgen -G ${stage1_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage1_dir}/src.de \
          model.train_ds.tgt_file_name=${stage1_dir}/tgt.en \
          model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
          model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
          model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
          model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
          exp_manager.exp_dir=${stage1_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=30000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=/workspace/old_results/result_de_en/best.ckpt
     fi \
  && export stage2_dir=${result_dir}/stage2 \
  && if [ ! -f ${stage2_dir}/best.ckpt ]; then
      cat ${bi_path}/train.clean.de ${mono_data_path}/6M_en_mono_partiaally_translated_06.12.20/translations.txt > ${stage2_dir}/src.de \
      && cat ${bi_path}/train.clean.en ${mono_data_path}/6M_en_mono_partiaally_translated_06.12.20/originals.txt > ${stage2_dir}/tgt.en \
      && if compgen -G ${stage2_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage2_dir}/src.de \
          model.train_ds.tgt_file_name=${stage2_dir}/tgt.en \
          model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
          model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
          model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
          model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
          exp_manager.exp_dir=${stage2_dir} \
          +exp_manager.resume_if_exists=${resume} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=50000 \
          +model.weights_checkpoint=${stage1_dir}/best.ckpt
     fi \
  && export stage3_dir=${result_dir}/stage3 \
  && if [ ! -f ${stage3_dir}/best.ckpt ]; then
      cat ${bi_path}/train.clean.de ${mono_data_path}/12M_en_mono_partiaally_translated_06.12.20/translations.txt > ${stage3_dir}/src.de \
      && cat ${bi_path}/train.clean.en ${mono_data_path}/12M_en_mono_partiaally_translated_06.12.20/originals.txt > ${stage3_dir}/tgt.en \
      && if compgen -G ${stage3_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage3_dir}/src.de \
          model.train_ds.tgt_file_name=${stage3_dir}/tgt.en \
          model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
          model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
          model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
          model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
          exp_manager.exp_dir=${stage3_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=75000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage2_dir}/best.ckpt
     fi \
  && export stage4_dir=${result_dir}/stage4 \
  && if [ ! -f ${stage4_dir}/best.ckpt ]; then
      cat ${bi_path}/train.clean.de ${mono_data_path}/24M_en_mono_partiaally_translated_06.12.20/translations.txt > ${stage4_dir}/src.de \
      && cat ${bi_path}/train.clean.en ${mono_data_path}/24M_en_mono_partiaally_translated_06.12.20/originals.txt > ${stage4_dir}/tgt.en \
      && if compgen -G ${stage4_dir}/TransformerMT/*/checkpoints/* > dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && python3 train.py -cn ${base_conf} \
          trainer.gpus=${ngpus} \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage4_dir}/src.de \
          model.train_ds.tgt_file_name=${stage4_dir}/tgt.en \
          model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
          model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
          model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
          model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
          exp_manager.exp_dir=${stage4_dir} \
          trainer.max_epochs=${max_epochs} \
          trainer.max_steps=100000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage3_dir}/best.ckpt
     fi \
  && python3 test.py -cn ${base_conf} \
      trainer.gpus=${ngpus} \
      model.train_ds.src_file_name=${stage4_dir}/src.de \
      model.train_ds.tgt_file_name=${stage4_dir}/tgt.en \
      model.validation_ds.src_file_name=${bi_path}/wmt13-de-en.src \
      model.validation_ds.tgt_file_name=${bi_path}/wmt13-de-en.ref \
      model.test_ds.src_file_name=${bi_path}/wmt14-de-en.src \
      model.test_ds.tgt_file_name=${bi_path}/wmt14-de-en.ref \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${stage4_dir}/best.ckpt