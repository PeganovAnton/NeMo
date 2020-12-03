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
  && export output=${train_results_path}/mono_en_translated.txt \
  && export best_ckpt=${train_results_path}/best.ckpt \
  && cat ${bi_path}/train.clean.* ${mono_en_path}/monolingual.2* ${mono_de_path}/monolingual.2* > all_text.txt \
  && yttm bpe --data all_text.txt --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && python nmt_transformer_infer.py \
      --model ${best_ckpt} \
      --text2translate ${mono_en_path}/monolingual.2* \
      --output ${output}