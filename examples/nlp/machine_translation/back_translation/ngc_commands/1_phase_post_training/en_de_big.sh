#!/bin/bash
# The only argument is wandb token.

WANDB_PROJECT=nmt_en_de_back_translation_4_phases_v0
TRANSLATE_MODELS_WS=wmt_translate_models
TRANSLATE_MODELS_PATH=/translate_models
BACK_TRANSLATION_DS_ID=71127
DATA_PATH=/data
BACK_TRANSLATED_PATH=${DATA_PATH}/backtranslated

read -r -d '' cmd <<EOF
set -e -x
git clone https://github.com/PeganovAnton/NeMo.git
cd NeMo
git checkout nmtmodel
pip3 install -r requirements/requirements.txt
pip3 install -r requirements/requirements_nlp.txt
pip3 install --upgrade wandb
pip3 install webdataset
wandb login $1
export nemo_path="\$(pwd)"
export HYDRA_FULL_ERROR=1
echo "NeMo path: \${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd  "\${nemo_path}/examples/nlp/machine_translation"
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

base_conf=wmt16/en_de_8gpu
train_n_tokens_in_batch=16000
max_epochs=100000
data_path=/data
phase_data_path=${data_path}/4_stages_back_translation_en_de
valid_src=${data_path}/parallel_clean/newstest2013-en-de.src
valid_ref=${data_path}/parallel_clean/newstest2013-en-de.ref
test_src=${data_path}/parallel_clean/newstest2014-en-de.src
test_ref=${data_path}/parallel_clean/newstest2014-en-de.ref
result_dir=/result
tok_model=${data_path}/bpe_35k_en_de_yttm.model
phase0_base_lr=0.0005
back_translation_phases_base_lr=0.00005
lr_num_nodes_factor=$(python -c "print(${SLURM_NUM_NODES}**0.5)")
echo "Learning rate number of nodes factor ${lr_num_nodes_factor}"
phase0_lr=$(python -c "print(${phase0_base_lr} * ${lr_num_nodes_factor})")
back_translation_phases_lr=$(python -c "print(${back_translation_phases_base_lr} * ${lr_num_nodes_factor})")
encoder_bpe_dropout=0.1
decoder_bpe_dropout=0.1

phase1_dir=${result_dir}/phase1
phase1_train_data="${phase_data_path}/phase1"
max_steps1=100000

mkdir -p ${phase1_dir} \
if [ ! -L ${phase1_dir}/best.ckpt ]; then
  if compgen -G ${phase1_dir}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 1 training from checkpoint"
    resume1=true
  else
    echo "Checkpoints for PHASE 1 are not found. Training will be started from scratch."
    resume1=false
  fi
  python enc_dec_nmt.py --config-name=aayn_base \
    trainer.gpus=${num_gpus} \
    ~trainer.max_epochs \
    +trainer.max_steps=${max_steps1}  \
    model.encoder_tokenizer.tokenizer_model=${tok_model}  \
    model.encoder_tokenizer.bpe_dropout=${encoder_bpe_dropout} \
    model.decoder_tokenizer.tokenizer_model=${tok_model}  \
    model.decoder_tokenizer.bpe_dropout=${decoder_bpe_dropout} \
    model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
    model.train_ds.src_file_name=${phase1_train_data}/originals.txt \
    model.train_ds.tgt_file_name=${phase1_train_data}/translations.txt \
    model.validation_ds.src_file_name=${valid_src} \
    model.validation_ds.tgt_file_name=${valid_ref} \
    model.test_ds.src_file_name=${test_src} \
    model.test_ds.tgt_file_name=${test_ref} \
    model.optim.lr=${phase1_lr}  \
    exp_manager.wandb_logger_kwargs.name=phase1 \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    +exp_manager.exp_dir=${phase1_dir} \
    +exp_manager.resume_if_exists=${resume1}
  echo  "*******FINISHING 1TH PHASE********"
fi

wmt13_translated=/result/newstest2013_en_translated.txt
python3 nmt_transformer_infer.py \
  --model ${phase1_dir}/best.ckpt \
  --srctext /data/wmt13_en_de.src \
  --tgtout "\${wmt13_translated}" \
  --target_lang de
cat "\${wmt13_translated}" | sacrebleu -t wmt13 -l en-de
wmt14_translated=/result/newstest2014_en_translated.txt
python3 nmt_transformer_infer.py \
  --model ${phase1_dir}/best.ckpt \
  --srctext /data/wmt14_en_de.src \
  --tgtout "\${wmt14_translated}" \
  --target_lang de
cat "\${wmt14_translated}" | sacrebleu -t wmt14 -l en-de
EOF

ngc batch run \
  --name big_en_de_attention_is_all_you_need_close_to_paper \
  --preempt RUNONCE \
  --image nvidia/pytorch:20.09-py3 \
  --ace nv-us-west-2 \
  --instance dgx1v.32g.8.norm \
  --commandline "${cmd}" \
  --result /result \
  --org nvidian \
  --team ac-aiapps \
  --datasetid ${BACK_TRANSLATION_DS_ID}:${BACK_TRANSLATED_PATH} \
  --workspace ${TRANSLATE_MODELS}:${TRANSLATE_MODELS_PATH}

