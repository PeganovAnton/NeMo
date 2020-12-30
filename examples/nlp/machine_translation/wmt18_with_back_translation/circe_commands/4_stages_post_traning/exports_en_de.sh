export nemo_path=/workspace/NeMo \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && base_conf=wmt16/de_en_8gpu \
  && num_nodes="$1" \
  && ngpus="$2" \
  && train_n_tokens_in_batch=11000 \
  && max_epochs=1000 \
  && data_path=/workspace/mydatasets/wmt18/4_stages_back_translation_en_de \
  && result_dir=/workspace/result_en_de_back_translation_4_stages_1st_trial \
  && tok_model=${data_path}/bpe_35k_en_de_yttm.model \
  && stage0_base_lr=0.001 \
  && back_translation_stages_base_lr=0.0001 \
  && lr_num_nodes_factor=$(( sqrt(${num_nodes}) )) \
  && stage0_lr=$(( ${stage0_base_lr} * ${lr_num_nodes_factor} )) \
  && back_translation_stages_lr=$(( ${back_translation_stages_base_lr} * ${lr_num_nodes_factor} ))
