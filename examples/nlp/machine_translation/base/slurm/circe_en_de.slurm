#!/bin/bash
#SBATCH -A dl-langspeech
#SBATCH -p batch
#SBATCH -N 1                    # number of nodes
#SBATCH -t 8:00:00              # wall time
#SBATCH -J "mem_tokens_for_mem_tokens_project"   # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node 16     # n tasks per machine (one task per GPU)
#SBATCH --cores-per-socket 24   # 24 cores per socket
#SBATCH --threads-per-core 2    # hyperthreading on
#SBATCH --sockets-per-node 2
#SBATCH --signal=SIGUSR1@90

# Arguments:
# WANDB_TOKEN -- a token for logging into wandb
# DATASET_NAME -- is a name of a directory with data in /gpfs/fs1/apeganov/datasets
# LAUNCH_NAME -- experiment. It may contain dataset description, number of iterations etc. It is used as name for
#   wandb logging
# TOKENIZER_DIR -- a path to a folder with tokenizer model
# TOKENIZER_NAME -- a name of a file with tokenizer model
# lr -- learning rate

set -x -e

WANDB_PROJECT=mem_tokens
CONTAINER="rumpelschtilzchen/pytorch-20.11-py3-nemo-nmt-mem-tokens:latest"
# Directories for manifests, data, etc.
DATASETS_PATH=/gpfs/fs1/apeganov/datasets
if [ -z ${DATASET_NAME} ]; then
  DATASET_NAME=wmt20_biclean60
fi
if [ -z ${LAUNCH_NAME} ]; then
  LAUNCH_NAME=mem_tokens_with_mem_tokens
fi
DATA_DIR=${DATASETS_PATH}/${DATASET_NAME}
data_dir=/data
result_dir=/result
if [ -z ${TOKENIZER_DIR} ]; then
  TOKENIZER_DIR=/gpfs/fs1/apeganov/models/mt/translation_pretrained_weights/large_en_de
fi
if [ -z ${TOKENIZER_NAME} ]; then
  TOKENIZER_NAME=tokenizer.latest.60.32000.BPE.model
fi
tokenizer_dir=/tokenizer_directory
EXP_NAME=${WANDB_PROJECT}
launch_prefix=${LAUNCH_NAME}_launch
GPFS=/gpfs/fs1/apeganov/results
if [ -z ${launch_number} ]; then
  launch_number=0
  while [ -e ${GPFS}/${EXP_NAME}/${launch_prefix}${launch_number} ]; do
    ((++launch_number))
  done
fi
launch_name=${launch_prefix}${launch_number}
RESULT_DIR=${GPFS}/${EXP_NAME}/${launch_name}
mkdir -p ${RESULT_DIR}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
MOUNTS="--container-mounts=${DATA_DIR}:${data_dir},${RESULT_DIR}:${result_dir},${TOKENIZER_DIR}:${tokenizer_dir}"
# Set up multinode if applicable
if [[ -z ${SLURM_JOB_ID} ]]; then
  hosts=( `hostname` )
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
fi

BATCH_SIZE=11000

text_data=${data_dir}/text
TARRED_DIR=tarred_16k
tarred_data=${data_dir}/${TARRED_DIR}
train_tar_files="${tarred_data}/train/batches.tokens.16000._OP_1..\$(ls ${data_dir}/${TARRED_DIR}/train/*.tar | wc -l)_CL_.tar"
train_metadata=${tarred_data}/train/metadata.json
valid_src=${text_data}/newstest2013.en
valid_ref=${text_data}/newstest2013.de
test_src=${text_data}/newstest2014-en-de.en
test_ref=${text_data}/newstest2014-en-de.de
if [ -z ${lr} ]; then
  base_lr=0.00015
  lr_num_nodes_factor=$(python3 -c "print(${SLURM_NNODES}**0.5)")
  echo "Learning rate number of nodes factor ${lr_num_nodes_factor}"
  lr=$(python3 -c "print(${base_lr} * ${lr_num_nodes_factor})")
fi
echo "Initial learning rate is ${lr}"
encoder_bpe_dropout=0.1
decoder_bpe_dropout=0.1

max_steps=120000

wmt13_translated=${result_dir}/newstest2013_en_translated.txt
wmt14_translated=${result_dir}/newstest2014_en_translated.txt

OUTFILE="${RESULT_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULT_DIR}/error-%j-%n.out"

val_check_interval=$(python3 -c "i=int(round(1000 / ${SLURM_JOB_NUM_NODES} / 100) * 100);print(i if i > 0 else 10)")

# Diagnostic prints
echo "Hosts: $hosts"
# Command to be run
echo "Your W&B token is ${WANDB_TOKEN}"

if compgen -G ${RESULT_DIR}/AAYNLarge/checkpoints/* > /dev/null; then
  echo "Will try to resume training from checkpoint"
  resume=true
else
  echo "Checkpoints are not found. Training will be started from scratch."
  resume=false
fi
set +e
read -r -d '' cmd <<EOF
set -x -e
echo "*******STARTING TRAINING********"
echo "---------------"
if [ "\${SLURM_PROCID}" -eq "0" ]; then rm -rf ~/.netrc; wandb login ${WANDB_TOKEN}; fi
nemo_path=/workspace/NeMo
echo "NeMo path: \${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd "\${nemo_path}/examples/nlp/machine_translation"
num_gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python train.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  trainer.gpus=\${num_gpus} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps} \
  +trainer.progress_bar_refresh_rate=0 \
  ~trainer.check_val_every_n_epoch \
  +trainer.val_check_interval=${val_check_interval} \
  model.encoder_tokenizer.tokenizer_model=${tokenizer_dir}/${TOKENIZER_NAME}  \
  model.encoder_tokenizer.bpe_dropout=${encoder_bpe_dropout} \
  model.decoder_tokenizer.tokenizer_model=${tokenizer_dir}/${TOKENIZER_NAME}  \
  model.decoder_tokenizer.bpe_dropout=${decoder_bpe_dropout} \
  +model.num_mem_tokens=32 \
  model.train_ds.tar_files=${train_tar_files} \
  model.train_ds.use_tarred_dataset=true \
  model.train_ds.metadata_file=${train_metadata} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${lr}  \
  exp_manager.wandb_logger_kwargs.name=circe_${launch_name} \
  exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
  +exp_manager.explicit_log_dir=${result_dir} \
  +exp_manager.resume_if_exists=${resume} \
  +model.find_unused_parameters=true
echo  "*******FINISHING TRAINING********"
set +x +e
EOF

read -r -d '' test_cmd <<EOF
set -x -e
if [ "\${SLURM_PROCID}" -eq "0" ]; then
  echo "*******STARTING TESTING ON %s********"
  echo "---------------"
  nemo_path=/workspace/NeMo
  echo "NeMo path: \${nemo_path}"
  export PYTHONPATH="\${nemo_path}"
  cd "\${nemo_path}/examples/nlp/machine_translation"
  translated_file=${result_dir}/%s
  python3 nmt_transformer_infer.py \
    --model %s \
    --srctext ${data_dir}/%s \
    --tgtout \${translated_file} \
    --target_lang de
  cat \${translated_file} | sacrebleu -t %s -l en-de | tee ${result_dir}/%s
  echo  "*******END TESTING ON %s*******"
fi
set +e +x
EOF

set -e

srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"


trained_model=${result_dir}/checkpoints/AAYNLarge.nemo
TEST_SRC_FILE=text/newstest2013.en
TEST_SCORE=sacreBLEU-newstest2013-after
TEST_TRANSLATED_PATH=newstest2013-translated-after.txt
printf -v test_cmd_run "${test_cmd}" \
  "${TEST_SRC_FILE}" \
  "${TEST_TRANSLATED_PATH}" \
  "${trained_model}" \
  "${TEST_SRC_FILE}" \
  wmt13 \
  "${TEST_SCORE}" \
  "${TEST_SRC_FILE}"
echo ${test_cmd_run}
srun -o ${RESULT_DIR}/slurm-newstest2013-after-training-%j-%n.out \
  -e ${RESULT_DIR}/error-newstest2013-after-training-%j-%n.out \
  --container-image="$CONTAINER" \
  -N 1-1 \
  $MOUNTS \
  bash -c "${test_cmd_run}"


TEST_SRC_FILE=text/newstest2014-en-de.en
TEST_SCORE=sacreBLEU-newstest2014-after
TEST_TRANSLATED_PATH=newstest2014-translated-after.txt
printf -v test_cmd_run "${test_cmd}" \
  "${TEST_SRC_FILE}" \
  "${TEST_TRANSLATED_PATH}" \
  "${trained_model}" \
  "${TEST_SRC_FILE}" \
  wmt14 \
  "${TEST_SCORE}" \
  "${TEST_SRC_FILE}"
echo ${test_cmd_run}
srun -o ${RESULT_DIR}/slurm-newstest2014-after-training-%j-%n.out \
  -e ${RESULT_DIR}/error-newstest2014-after-training-%j-%n.out \
  --container-image="$CONTAINER" \
  -N 1-1 \
  $MOUNTS \
  bash -c "${test_cmd_run}"
set +x +e

