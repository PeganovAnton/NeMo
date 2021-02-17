#!/bin/bash
# The only argument is wandb token.

WANDB_PROJECT=1_phase_post_training_on_sandeep_back_translated_data
TRANSLATE_MODELS_WS=trainslation_pretrained_weights
TRANSLATE_MODELS_PATH=/wmt_translate_models
DS_ID=74337
DATA_PATH=/data

TRAIN_N_TOKENS_IN_BATCH=16000
MAX_EPOCHS=100000
MAX_STEPS=100000
PARALLEL_PATH=${DATA_PATH}/text
TRAIN_SRC=${PARALLEL_PATH}/train.en
TRAIN_REF=${PARALLEL_PATH}/train.de
VALID_SRC=${PARALLEL_PATH}/newstest2013.en
VALID_REF=${PARALLEL_PATH}/newstest2013.de
TEST_SRC=${PARALLEL_PATH}/newstest2014.en
TEST_REF=${PARALLEL_PATH}/newstest2014.de
RESULT_DIR=/result
PRETRAINED_PATH=${TRANSLATE_MODELS_PATH}/large_en_de
TOK_MODEL=${PRETRAINED_PATH}/tokenizer.latest.60.32000.BPE.model
BASE_LR=0.0005
ENCODER_BPE_DROPOUT=0.1
DECODER_BPE_DROPOUT=0.1
EXP_DIR=${RESULT_DIR}/exp_dir

WMT13_TRANSLATED=${RESULT_DIR}/newstest2013_en_translated.txt
WMT14_TRANSLATED=${RESULT_DIR}/newstest2014_en_translated.txt

read -r -d '' cmd <<EOF
set -e -x
git clone https://github.com/PeganovAnton/NeMo.git
cd NeMo
git checkout sacreBLEU_pl_metric
pip3 install -r requirements/requirements.txt
pip3 install -r requirements/requirements_nlp.txt
pip3 install --upgrade wandb
pip3 install webdataset
wandb login $1
export nemo_path="\$(pwd)"
export HYDRA_FULL_ERROR=1
echo "NeMo path: \${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd "\${nemo_path}/examples/nlp/machine_translation"
num_gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


python train.py --config-name=aayn_big \
  trainer.gpus=\${num_gpus} \
  ~trainer.max_epochs \
  +trainer.max_steps=${MAX_STEPS}  \
  +model.weights_checkpoin=${PRETRAINED_PATH}/model_weights.ckpt \
  model.encoder_tokenizer.tokenizer_model=${TOK_MODEL}  \
  model.encoder_tokenizer.bpe_dropout=${ENCODER_BPE_DROPOUT} \
  model.decoder_tokenizer.tokenizer_model=${TOK_MODEL}  \
  model.decoder_tokenizer.bpe_dropout=${DECODER_BPE_DROPOUT} \
  model.train_ds.tokens_in_batch=${TRAIN_N_TOKENS_IN_BATCH} \
  model.train_ds.src_file_name=${TRAIN_SRC} \
  model.train_ds.tgt_file_name=${TRAIN_REF} \
  model.validation_ds.src_file_name=${VALID_SRC} \
  model.validation_ds.tgt_file_name=${VALID_REF} \
  model.test_ds.src_file_name=${TEST_SRC} \
  model.test_ds.tgt_file_name=${TEST_REF} \
  model.optim.lr=${BASE_LR}  \
  exp_manager.wandb_logger_kwargs.name=1st_trial \
  exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
  +exp_manager.exp_dir=${EXP_DIR}

python3 nmt_transformer_infer.py \
  --model ${phase1_dir}/best.ckpt \
  --srctext /data/wmt13_en_de.src \
  --tgtout "${WMT13_TRANSLATED}" \
  --target_lang de
cat "${WMT13_TRANSLATED}" | sacrebleu -t wmt13 -l en-de

python3 nmt_transformer_infer.py \
  --model ${phase1_dir}/best.ckpt \
  --srctext /data/wmt14_en_de.src \
  --tgtout "${WMT14_TRANSLATED}" \
  --target_lang de
cat "${WMT14_TRANSLATED}" | sacrebleu -t wmt14 -l en-de
set +e +x
EOF

ngc batch run \
  --name big_en_de_attention_is_all_you_need_close_to_paper \
  --preempt RUNONCE \
  --image nvidia/pytorch:20.11-py3 \
  --ace nv-us-west-2 \
  --instance dgx1v.16g.8.norm \
  --commandline "${cmd}" \
  --result /result \
  --org nvidian \
  --team ac-aiapps \
  --datasetid ${DS_ID}:${DATA_PATH} \
  --workspace ${TRANSLATE_MODELS_WS}:${TRANSLATE_MODELS_PATH}:RO

