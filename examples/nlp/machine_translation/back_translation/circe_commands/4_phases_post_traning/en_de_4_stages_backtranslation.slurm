#!/bin/bash
#SBATCH -A dl-langspeech
#SBATCH -p batch
#SBATCH -N 8                    # number of nodes
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
CONTAINER="rumpelschtilzchen/pytorch-20.11-py3-nemo-nmt:latest"
# Directories for manifests, data, etc.
DATA_DIR='/gpfs/fs1/apeganov/datasets/wmt18'
RESULT_DIR='/home/apeganov'
EXP_NAME='result_en_de_back_translation_4_stages_1st_trial'
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
mkdir -p $RESULT_DIR/$EXP_NAME
MOUNTS="--container-mounts=$DATA_DIR:/data,$RESULT_DIR/$EXP_NAME:/result"
# Set up multinode if applicable
if [[ -z "$SLURM_JOB_ID" ]]; then
  hosts=( `hostname` )
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
fi

SLURM_NUM_NODES=8
SLURM_NUM_GPUS=16

BATCH_SIZE=11000

base_conf=wmt16/en_de_8gpu
train_n_tokens_in_batch=11000
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

phase0_dir=${result_dir}/phase0
PHASE0_DIR=${RESULT_DIR}/${EXP_NAME}/phase0
phase0_train_data="${phase_data_path}/phase0"
max_steps0=60000
OUTFILE0="${RESULT_DIR}/${EXP_NAME}/slurm-phase0-%j-%n.out"
ERRFILE0="${RESULT_DIR}/${EXP_NAME}/error-phase0-%j-%n.out"

phase1_dir=${result_dir}/phase1
PHASE1_DIR=${RESULT_DIR}/${EXP_NAME}/phase1
phase1_train_data="${phase_data_path}/phase1"
max_steps1=100000
OUTFILE1="${RESULT_DIR}/${EXP_NAME}/slurm-phase1-%j-%n.out"
ERRFILE1="${RESULT_DIR}/${EXP_NAME}/error-phase1-%j-%n.out"

phase2_dir=${result_dir}/phase2
PHASE2_DIR=${RESULT_DIR}/${EXP_NAME}/phase2
phase2_train_data="${phase_data_path}/phase2"
max_steps2=150000
OUTFILE2="${RESULT_DIR}/${EXP_NAME}/slurm-phase2-%j-%n.out"
ERRFILE2="${RESULT_DIR}/${EXP_NAME}/error-phase2-%j-%n.out"

phase3_dir=${result_dir}/phase3
PHASE3_DIR=${RESULT_DIR}/${EXP_NAME}/phase3
phase3_train_data="${phase_data_path}/phase3"
max_steps3=150000
OUTFILE3="${RESULT_DIR}/${EXP_NAME}/slurm-phase3-%j-%n.out"
ERRFILE3="${RESULT_DIR}/${EXP_NAME}/error-phase3-%j-%n.out"

phase4_dir=${result_dir}/phase4
PHASE4_DIR=${RESULT_DIR}/${EXP_NAME}/phase4
phase4_train_data="${phase_data_path}/phase4"
max_steps4=200000
OUTFILE4="${RESULT_DIR}/${EXP_NAME}/slurm-phase4-%j-%n.out"
ERRFILE4="${RESULT_DIR}/${EXP_NAME}/error-phase4-%j-%n.out"

val_check_interval=$(python -c "i=int(round(1000 / ${SLURM_NUM_NODES} / 100) * 100);print(i if i > 0 else 10)")

# Diagnostic prints
echo "Hosts: $hosts"
# Command to be run
echo "Your W&B token is ${WANDB_TOKEN}"

mkdir -p ${PHASE0_DIR} \
&& if [ ! -L ${PHASE0_DIR}/best.ckpt ]; then
  if compgen -G ${PHASE0_DIR}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 0 training from checkpoint"
    resume0=true
  else
    echo "Checkpoints for PHASE 0 are not found. Training will be started from scratch."
    resume0=false
  fi \
  && read -r -d '' cmd0 <<EOF
echo "*******STARTING 0TH PHASE********" \
&& echo "---------------" \
&& set -x \
&& wandb login ${WANDB_TOKEN} \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python enc_dec_nmt.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_NUM_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps0}  \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase0_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase0_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${phase0_lr}  \
  exp_manager.wandb_logger_kwargs.name=phase0 \
  exp_manager.wandb_logger_kwargs.project=nmt_en_de_back_translation_4_phases_v0 \
  exp_manager.exp_dir=${phase0_lr} \
  +exp_manager.resume_if_exists=${resume0} \
&& set + \
&& echo  "*******FINISHING 0TH PHASE********"
EOF
  srun -o $OUTFILE0 -e $ERRFILE0 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd0}"
fi \
&& mkdir -p ${PHASE1_DIR} \
&& if [ ! -L ${PHASE1_DIR}/best.ckpt ]; then
  if compgen -G ${PHASE1_DIR}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 1 training from checkpoint"
    resume1=true
  else
    echo "Checkpoints for PHASE 1 are not found. Training will be started from scratch."
    resume1=false
  fi \
  && read -r -d '' cmd1 <<EOF
echo "*******STARTING 1ST PHASE********" \
&& echo "---------------" \
&& set -x \
&& wandb login ${WANDB_TOKEN} \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python enc_dec_nmt.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_NUM_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps1}  \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase1_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase1_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${phase1_lr}  \
  exp_manager.wandb_logger_kwargs.name=phase1 \
  exp_manager.wandb_logger_kwargs.project=nmt_en_de_back_translation_4_phases_v0 \
  exp_manager.exp_dir=${phase1_lr} \
  +exp_manager.resume_if_exists=${resume1} \
&& set + \
&& echo  "*******FINISHING 1ST PHASE********"
EOF
  srun -o $OUTFILE1 -e $ERRFILE1 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd1}"
fi \
&& mkdir -p ${PHASE2_DIR} \
&& if [ ! -L ${PHASE2_DIR}/best.ckpt ]; then
  if compgen -G ${PHASE2_DIR}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 2 training from checkpoint"
    resume2=true
  else
    echo "Checkpoints for PHASE 2 are not found. Training will be started from scratch."
    resume2=false
  fi \
  && read -r -d '' cmd2 <<EOF
echo "*******STARTING 2ND PHASE********" \
&& echo "---------------" \
&& set -x \
&& wandb login ${WANDB_TOKEN} \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python enc_dec_nmt.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_NUM_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps2}  \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase2_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase2_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${phase2_lr}  \
  exp_manager.wandb_logger_kwargs.name=phase2 \
  exp_manager.wandb_logger_kwargs.project=nmt_en_de_back_translation_4_phases_v0 \
  exp_manager.exp_dir=${phase2_lr} \
  +exp_manager.resume_if_exists=${resume2} \
&& set + \
&& echo  "*******FINISHING 2ND PHASE********"
EOF
  srun -o $OUTFILE2 -e $ERRFILE2 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd2}"
fi \
&& mkdir -p ${PHASE3_DIR} \
&& if [ ! -L ${PHASE3_DIR}/best.ckpt ]; then
  if compgen -G ${PHASE3_DIR}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 3 training from checkpoint"
    resume3=true
  else
    echo "Checkpoints for PHASE 3 are not found. Training will be started from scratch."
    resume3=false
  fi \
  && read -r -d '' cmd3 <<EOF
echo "*******STARTING 3RD PHASE********" \
&& echo "---------------" \
&& set -x \
&& wandb login ${WANDB_TOKEN} \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python enc_dec_nmt.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_NUM_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps3}  \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase3_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase3_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${phase3_lr}  \
  exp_manager.wandb_logger_kwargs.name=phase3 \
  exp_manager.wandb_logger_kwargs.project=nmt_en_de_back_translation_4_phases_v0 \
  exp_manager.exp_dir=${phase3_lr} \
  +exp_manager.resume_if_exists=${resume3} \
&& set + \
&& echo  "*******FINISHING 3RD PHASE********"
EOF
  srun -o $OUTFILE3 -e $ERRFILE3 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd3}"
fi \
&& mkdir -p ${PHASE4_DIR} \
&& if [ ! -L ${PHASE4_DIR}/best.ckpt ]; then
  if compgen -G ${PHASE4_DIR}/TransformerMT/checkpoints/* > /dev/null; then
    echo "Will try to resume PHASE 4 training from checkpoint"
    resume4=true
  else
    echo "Checkpoints for PHASE 4 are not found. Training will be started from scratch."
    resume4=false
  fi \
  && read -r -d '' cmd4 <<EOF
echo "*******STARTING 4TH PHASE********" \
&& echo "---------------" \
&& set -x \
&& wandb login ${WANDB_TOKEN} \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python enc_dec_nmt.py --config-name=aayn_base \
  trainer.num_nodes=${SLURM_NUM_NODES} \
  trainer.gpus=${SLURM_NUM_GPUS} \
  ~trainer.max_epochs \
  +trainer.max_steps=${max_steps4}  \
  model.encoder_tokenizer.tokenizer_model=${tok_model}  \
  model.decoder_tokenizer.tokenizer_model=${tok_model}  \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase4_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase4_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.optim.lr=${phase4_lr}  \
  exp_manager.wandb_logger_kwargs.name=phase4 \
  exp_manager.wandb_logger_kwargs.project=nmt_en_de_back_translation_4_phases_v0 \
  exp_manager.exp_dir=${phase4_lr} \
  +exp_manager.resume_if_exists=${resume4} \
&& set + \
&& echo  "*******FINISHING 4TH PHASE********"
EOF
  srun -o $OUTFILE4 -e $ERRFILE4 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd4}"
fi

set +
