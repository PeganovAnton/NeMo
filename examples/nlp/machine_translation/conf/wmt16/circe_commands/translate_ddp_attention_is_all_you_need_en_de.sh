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
  && export train_results_path=/workspace/old_results/result_en_de \
  && export translations_file=${train_results_path}/mono_en_translated_to_de.txt \
  && export result_dir=${train_results_path}/mono_en_translation \
  && export best_ckpt=${train_results_path}/best.ckpt \
  && export yttm_model=bpe_16k_en_de_yttm.model \
  && if [ ! -f ${yttm_model} ]; then cat ${bi_path}/train.clean.en ${bi_path}/train.clean.de ${mono_en_path}/monolingual.25000000.en ${mono_de_path}/monolingual.25000000.de > all_text.txt && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000; fi \
  && python translate_ddp.py \
      --model ${best_ckpt} \
      --text2translate ${mono_en_path}/monolingual.25000000.en \
      --tokenizer_model bpe_16k_en_de_yttm.model \
      --max_num_tokens_in_batch 16000 \
      --result_dir ${result_dir}
