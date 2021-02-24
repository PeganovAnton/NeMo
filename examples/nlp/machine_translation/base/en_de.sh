#!/bin/bash
# The only argument is wandb token.

WANDB_PROJECT=mem_tokens
LAUNCH_NAME=$2
TRANSLATE_MODELS_WS=trainslation_pretrained_weights
TRANSLATE_MODELS_PATH=/wmt_translate_models
DS_ID=74702
DATA_PATH=/data

MAX_EPOCHS=100000
MAX_STEPS=100000
TEXT_PATH=${DATA_PATH}/text
TARRED_PATH=${DATA_PATH}/tarred_8k
RAID=/raid
RAID_TRAIN_PATH=${RAID}/train
TRAIN_TAR_FILES="${RAID_TRAIN_PATH}/batches.tokens.16000._OP_1..\$(ls ${TARRED_PATH}/train/*.tar | wc -l)_CL_.tar"
echo ${TRAIN_TAR_FILES}
TRAIN_METADATA=${RAID_TRAIN_PATH}/metadata.json
VALID_SRC=${RAID}/newstest2013.en
VALID_REF=${RAID}/newstest2013.de
TEST_SRC=${RAID}/newstest2014-en-de.en
TEST_REF=${RAID}/newstest2014-en-de.de
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
git checkout mem_tokens_new
pip3 install -r requirements/requirements.txt
pip3 install -r requirements/requirements_nlp.txt
pip3 install --upgrade wandb
pip3 install webdataset
pip3 install pyarrow
wandb login $1
export nemo_path="\$(pwd)"
export HYDRA_FULL_ERROR=1
echo "NeMo path: \${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd "\${nemo_path}/examples/nlp/machine_translation"
num_gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
cp -rv ${TEXT_PATH}/newstest* ${RAID}/
mkdir -p ${RAID_TRAIN_PATH}
cp -rv ${TARRED_PATH}/* ${RAID}/
mkdir -p ${EXP_DIR}
python train.py --config-name=aayn_base \
  trainer.gpus=\${num_gpus} \
  ~trainer.max_epochs \
  +trainer.max_steps=${MAX_STEPS}  \
  +trainer.progress_bar_refresh_rate=0 \
  +trainer.val_check_interval=200 \
  model.encoder_tokenizer.tokenizer_model=${TOK_MODEL}  \
  model.encoder_tokenizer.bpe_dropout=${ENCODER_BPE_DROPOUT} \
  model.decoder_tokenizer.tokenizer_model=${TOK_MODEL}  \
  model.decoder_tokenizer.bpe_dropout=${DECODER_BPE_DROPOUT} \
  +model.train_ds.tar_files=${TRAIN_TAR_FILES} \
  model.train_ds.use_tarred_dataset=true \
  +model.train_ds.metadata_file=${TRAIN_METADATA} \
  +model.num_mem_tokens=32 \
  model.validation_ds.src_file_name=${VALID_SRC} \
  model.validation_ds.tgt_file_name=${VALID_REF} \
  model.test_ds.src_file_name=${TEST_SRC} \
  model.test_ds.tgt_file_name=${TEST_REF} \
  model.optim.lr=${BASE_LR}  \
  +model.find_unused_parameters=true \
  exp_manager.wandb_logger_kwargs.name=${LAUNCH_NAME} \
  exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
  +exp_manager.exp_dir=${EXP_DIR}

python3 nmt_transformer_infer.py \
  --model ${EXP_DIR}/AAYNLarge/checkpoints/AAYNLarge.nemo \
  --srctext /data/wmt13_en_de.src \
  --tgtout "${WMT13_TRANSLATED}" \
  --target_lang de
cat "${WMT13_TRANSLATED}" | sacrebleu -t wmt13 -l en-de

python3 nmt_transformer_infer.py \
  --model ${EXP_DIR}/AAYNLarge/checkpoints/AAYNLarge.nemo \
  --srctext /data/wmt14_en_de.src \
  --tgtout "${WMT14_TRANSLATED}" \
  --target_lang de
cat "${WMT14_TRANSLATED}" | sacrebleu -t wmt14 -l en-de
set +e +x
EOF

ngc batch run \
  --name mem_tokens_in_transformer \
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

