#!/bin/bash
#SBATCH -A dl-langspeech
#SBATCH -p batch
#SBATCH -N 1                    # number of nodes
#SBATCH -t 8:00:00              # wall time
#SBATCH -J "en_de_4_staged_back_translation"   # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node 16    # n tasks per machine (one task per GPU)
#SBATCH --cores-per-socket 24   # 24 cores per socket
#SBATCH --threads-per-core 2    # hyperthreading on
#SBATCH --sockets-per-node 2
#SBATCH --signal=SIGUSR1@90
# --overcommit option was removed
# Hyper params
# Env Variable Setup
set -x

WANDB_PROJECT=1_phase_post_training_on_sandeep_back_translated_data
CONTAINER="rumpelschtilzchen/pytorch-20.11-py3-nemo-nmt:latest"
# Directories for manifests, data, etc.
DATA_DIR=/gpfs/fs1/apeganov/datasets/wmt20_en_de_with_back_translated
data_dir=/data
RESULT_DIR=/home/apeganov
result_dir=/result
PRETRAINED_MODELS=/gpfs/fs1/apeganov/models/mt/translation_pretrained_weights/large_en_de
pretrained_models=/pretrained
EXP_NAME=news_back_translation_training_1_phase
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
mkdir -p $RESULT_DIR/$EXP_NAME
MOUNTS="--container-mounts=$DATA_DIR:$data_dir,$RESULT_DIR/$EXP_NAME:${result_dir},${PRETRAINED_MODELS}:${pretrained_models}"
# Set up multinode if applicable
if [[ -z "$SLURM_JOB_ID" ]]; then
  hosts=( `hostname` )
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
fi

SLURM_NUM_GPUS=16

BATCH_SIZE=11000

max_epochs=100000
text_data=${data_dir}/text
tarred_data=${data_dir}/tarred
train_tar_files=${tarred_data}/train/batches.tokens.16000._OP_1..71_CL_.tar
train_metadata=${tarred_data}/train/metadata.json
valid_src=${text_data}/newstest2013.en
valid_ref=${text_data}/newstest2013.de
test_src=${text_data}/newstest2014.en
test_ref=${text_data}/newstest2014.de
tok_model=${pretrained_models}/tokenizer.latest.60.32000.BPE.model
phase0_base_lr=0.0005
back_translation_phases_base_lr=0.00005
lr_num_nodes_factor=$(python -c "print(${SLURM_JOB_NODES}**0.5)")
echo "Learning rate number of nodes factor ${lr_num_nodes_factor}"
lr=$(python -c "print(${phase0_base_lr} * ${lr_num_nodes_factor})")
back_translation_phases_lr=$(python -c "print(${back_translation_phases_base_lr} * ${lr_num_nodes_factor})")
encoder_bpe_dropout=0.1
decoder_bpe_dropout=0.1

EXP_DIR=${RESULT_DIR}/${EXP_NAME}
max_steps=100000
OUTFILE="${RESULT_DIR}/${EXP_NAME}/slurm-%j-%n.out"
ERRFILE="${RESULT_DIR}/${EXP_NAME}/error-%j-%n.out"

val_check_interval=$(python -c "i=int(round(1000 / ${SLURM_JOB_NODES} / 100) * 100);print(i if i > 0 else 10)")

# Diagnostic prints
echo "Hosts: $hosts"
# Command to be run
echo "Your W&B token is ${WANDB_TOKEN}"

if compgen -G ${EXP_DIR}/TransformerMT/checkpoints/* > /dev/null; then
  echo "Will try to resume training from checkpoint"
  resume=true
else
  echo "Checkpoints are not found. Training will be started from scratch."
  resume=false
fi
read -r -d '' cmd <<EOF
set -x -e
echo "*******STARTING TRAINING********"
echo "---------------"
set -x -e
if [ "\${SLURM_PROCID}" -eq "0" ]; then wandb login ${WANDB_TOKEN}; fi
nemo_path=/workspace/NeMo
echo "NeMo path: ${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd "\${nemo_path}/examples/nlp/machine_translation"
ls ${pretrained_models}
python train.py --config-name=aayn_big \
  trainer.num_nodes=${SLURM_JOB_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps}  \
  +model.weights_checkpoint=${pretrained_models}/model_weights.ckpt \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.encoder_tokenizer.bpe_dropout=${encoder_bpe_dropout} \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.bpe_dropout=${decoder_bpe_dropout} \
  +model.train_ds.tar_files=${train_tar_files} \
  +model.train_ds.use_tarred_dataset=true \
  +model.train_ds.metadata_file=${train_metadata} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${lr}  \
  exp_manager.wandb_logger_kwargs.name=circe_1st_trial \
  exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
  +exp_manager.exp_dir=${result_dir} \
  +exp_manager.resume_if_exists=${resume} \
  +model.find_unused_parameters=true
echo  "*******FINISHING TRAINING********"
set +x +e
EOF
srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"

set +x
