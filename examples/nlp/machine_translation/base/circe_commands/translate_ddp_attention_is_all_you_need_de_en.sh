pip3 install -r requirements/requirements.txt \
  && pip3 install -r requirements/requirements_nlp.txt \
  && pip3 install webdataset \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export mono_data_path=/workspace/mydatasets/apeganov/wmt18 \
  && export bi_path=/workspace/mydatasets/apeganov/wmt/wmt18_en_de/parallel \
  && export mono_en_path=${mono_data_path}/wmt18_en_mono \
  && export mono_de_path=${mono_data_path}/wmt18_de_mono \
  && export train_results_path=/workspace/old_results/result_de_en \
  && export result_dir=${train_results_path}/mono_de_translation \
  && export best_ckpt=${train_results_path}/best.ckpt \
  && export yttm_model=bpe_16k_en_de_yttm.model \
  && if [ ! -f ${yttm_model} ]; then cat ${bi_path}/train.clean.en ${bi_path}/train.clean.de ${mono_en_path}/monolingual.25000000.en ${mono_de_path}/monolingual.25000000.de > all_text.txt && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000; fi \
  && python3 translate_ddp.py \
      --model ${best_ckpt} \
      --text2translate ${mono_de_path}/monolingual.25000000.de \
      --tokenizer_model bpe_16k_en_de_yttm.model \
      --max_num_tokens_in_batch 4000 \
      --result_dir ${result_dir}
