pip3 install -r requirements/requirements.txt \
  && pip3 install -r requirements/requirements_nlp.txt \
  && pip3 install webdataset \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && export base_conf=wmt16/de_en_8gpu \
  && export num_gpus="$3" \
  && export num_nodes="$2" \
  && if [ "${num_nodes}" -gt 1 ]; then
    declare -a host_np_arr \
    && for i in $(seq 1 1 ${num_nodes}); do host_np_arr+=("hostname${i}:${num_gpus}"); done \
    && export host_np=$(printf ",%s" "${host_np_arr[@]}") \
    && export host_np=${host_np:1}
  else
    export host_np=""
  fi \
  && export total_np=$(( ${num_nodes}*${num_gpus} )) \
  && export train_n_tokens_in_batch=11000 \
  && export max_epochs=1000 \
  && export mono_data_path=/data/translated \
  && export bi_path=/data/wmt18_parallel \
  && export tok_model=${bi_path}/bpe_35k_en_de_yttm.model \
  && export result_dir=/result \
  && export stage1_dir=${result_dir}/stage1 \
  && export valid_src=${bi_path}/newstest2013-de-en.src \
  && export valid_ref=${bi_path}/newstest2013-de-en.ref \
  && export valid_src=${bi_path}/newstest2014-de-en.src \
  && export valid_ref=${bi_path}/newstest2014-de-en.ref \
  && export par_train_src=${bi_path}/train.clean.filter.de.shuffled \
  && export par_train_ref=${bi_path}/train.clean.filter.en.shuffled \
  && mkdir -p ${stage1_dir} \
  && if [ "$1" -eq "0" ]; then
      export stage0_dir=${result_dir}/stage0 \
      && mkdir -p ${stage0_dir} \
      && if [ ! -f ${stage0_dir}/best.ckpt ]; then
          cat ${par_train_src} >"${stage0_dir}/originals.txt" \
          && cat ${par_train_ref} >"${stage0_dir}/translations.txt" \
          && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage0_dir} \
          && if compgen -G ${stage0_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
            export resume=true
          else
            export resume=false
          fi \
          && horovodrun -np ${total_np} -H ${host_np} python3 train.py -cn ${base_conf} \
              trainer.num_nodes=${num_nodes} \
              trainer.gpus=${num_gpus} \
              trainer.accelerator=horovod \
              model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
              model.train_ds.src_file_name=${stage0_dir}/originals.txt \
              model.train_ds.tgt_file_name=${stage0_dir}/translations.txt \
              model.validation_ds.src_file_name=${valid_src} \
              model.validation_ds.tgt_file_name=${valid_ref} \
              model.test_ds.src_file_name=${test_src} \
              model.test_ds.tgt_file_name=${test_ref} \
              model.tokenizer.tokenizer_model=${tok_model} \
              exp_manager.exp_dir=${stage0_dir} \
              trainer.max_epochs=${max_epochs} \
              +trainer.max_steps=30000 \
              +exp_manager.resume_if_exists=${resume}
      fi \
      && export stage1_weights=${stage0_dir}/best.ckpt
    else
      export stage1_weights="$1"
    fi \
  && export stage1_dir=${result_dir}/stage1 \
  && mkdir -p ${stage1_dir} \
  && if [ ! -f ${stage1_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/en/3M/translations.txt >"${stage1_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/en/3M/originals.txt >"${stage1_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage1_dir} \
      && if compgen -G ${stage1_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && horovodrun -np ${total_np} -H ${host_np} python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${num_gpus} \
          trainer.accelerator=horovod \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage1_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage1_dir}/translations.txt \
          model.tokenizer.tokenizer_model=${tok_model} \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage1_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=50000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage1_weights}
    fi \
  && export stage2_dir=${result_dir}/stage2 \
  && mkdir -p ${stage2_dir} \
  && if [ ! -f ${stage2_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/en/6M/translations.txt >"${stage2_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/en/6M/originals.txt >"${stage2_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage2_dir} \
      && if compgen -G ${stage2_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && horovodrun -np ${total_np} -H ${host_np}  python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${num_gpus} \
          trainer.accelerator=horovod \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage2_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage2_dir}/translations.txt \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage2_dir} \
          +exp_manager.resume_if_exists=${resume} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=75000 \
          +model.weights_checkpoint=${stage1_dir}/best.ckpt \
          model.tokenizer.tokenizer_model=${tok_model}
     fi \
  && export stage3_dir=${result_dir}/stage3 \
  && mkdir -p ${stage3_dir} \
  && if [ ! -f ${stage3_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/en/12M/translations.txt >"${stage3_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/en/12M/originals.txt >"${stage3_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage3_dir} \
      && if compgen -G ${stage3_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && horovodrun -np ${total_np} -H ${host_np}  python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${num_gpus} \
          trainer.accelerator=horovod \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage3_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage3_dir}/translations.txt \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage3_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=75000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage2_dir}/best.ckpt \
          model.tokenizer.tokenizer_model=${tok_model}
     fi \
  && export stage4_dir=${result_dir}/stage4 \
  && mkdir -p ${stage4_dir} \
  && if [ ! -f ${stage4_dir}/best.ckpt ]; then
      cat ${par_train_src} ${mono_data_path}/en/24M/translations.txt >"${stage4_dir}/originals.txt" \
      && cat ${par_train_ref} ${mono_data_path}/en/24M/originals.txt >"${stage4_dir}/translations.txt" \
      && python3 ${nemo_path}/utils/shuffle_pairs.py ${stage4_dir} \
      && if compgen -G ${stage4_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
        export resume=true
      else
        export resume=false
      fi \
      && horovodrun -np ${total_np} -H ${host_np} python3 train.py -cn ${base_conf} \
          trainer.num_nodes=${num_nodes} \
          trainer.gpus=${num_gpus} \
          trainer.accelerator=horovod \
          model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
          model.train_ds.src_file_name=${stage4_dir}/originals.txt \
          model.train_ds.tgt_file_name=${stage4_dir}/translations.txt \
          model.validation_ds.src_file_name=${valid_src} \
          model.validation_ds.tgt_file_name=${valid_ref} \
          model.test_ds.src_file_name=${test_src} \
          model.test_ds.tgt_file_name=${test_ref} \
          exp_manager.exp_dir=${stage4_dir} \
          trainer.max_epochs=${max_epochs} \
          +trainer.max_steps=100000 \
          +exp_manager.resume_if_exists=${resume} \
          +model.weights_checkpoint=${stage3_dir}/best.ckpt \
          model.tokenizer.tokenizer_model=${tok_model}
     fi \
  && horovodrun -np ${total_np} -H ${host_np} python3 test.py -cn ${base_conf} \
      trainer.num_nodes=${num_nodes} \
      trainer.gpus=${num_gpus} \
      trainer.accelerator=horovod \
      model.train_ds.src_file_name=${stage4_dir}/originals.txt \
      model.train_ds.tgt_file_name=${stage4_dir}/translations.txt \
      model.validation_ds.src_file_name=${valid_src} \
      model.validation_ds.tgt_file_name=${valid_ref} \
      model.test_ds.src_file_name=${test_src} \
      model.test_ds.tgt_file_name=${test_ref} \
      exp_manager.exp_dir=${result_dir}/testing \
      model.test_checkpoint_path=${stage4_dir}/best.ckpt \
      model.tokenizer.tokenizer_model=${tok_model}