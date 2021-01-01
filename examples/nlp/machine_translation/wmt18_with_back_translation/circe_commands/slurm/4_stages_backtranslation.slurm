#!/bin/bash
#SBATCH -A dl-langspeech
#SBATCH -p batch
#SBATCH -N 2                    # number of nodes
#SBATCH -t 8:00:00              # wall time
#SBATCH -J "test_mul_node"   # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node 1     # n tasks per machine (one task per GPU)
#SBATCH --cores-per-socket 24   # 24 cores per socket
#SBATCH --threads-per-core 2    # hyperthreading on
#SBATCH --sockets-per-node 2
#SBATCH --overcommit            # Needed for pytorch
# Hyper params
# Env Variable Setup
set -x
CONTAINER="gitlab-master.nvidia.com/okuchaiev/nemo_containers:nemo-1.0.0a0"
# Directories for manifests, data, etc.
DATA_DIR='/gpfs/fs1/apeganov/datasets/wmt18/4_stages_back_translation_en_de'
RESULTS_DIR='/home/apeganov'
EXP_NAME='result_de_en_back_translation_4_stages_1st_trial'
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
mkdir -p $RESULTS_DIR/$EXP_NAME
MOUNTS="--container-mounts=$DATA_DIR:/data,$RESULTS_DIR/$EXP_NAME:/results"
# Set up multinode if applicable
SRUN_HOST=''
SRUN_CONT=''
if [[ -z "$SLURM_JOB_ID" ]]; then
  hosts=( `hostname` )
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
fi

SLURM_NUM_NODES=8
SLURM_NUM_GPUS=16

BASE_CONF=wmt16/de_en_8gpu
BATCH_SIZE=11000

base_conf=wmt16/de_en_8gpu
train_n_tokens_in_batch=11000
max_epochs=1000
data_path=/workspace/mydatasets/wmt18/4_phases_back_translation_en_de
result_dir=/workspace/result_en_de_back_translation_4_phases_1st_trial
tok_model=${data_path}/bpe_35k_en_de_yttm.model
phase0_base_lr=0.001
back_translation_phases_base_lr=0.0001
lr_num_nodes_factor=$(( sqrt(${num_nodes}) ))
phase0_lr=$(( ${phase0_base_lr} * ${lr_num_nodes_factor} ))
back_translation_phases_lr=$(( ${back_translation_phases_base_lr} * ${lr_num_nodes_factor} ))

phase0_dir=${result_dir}/phase0
phase0_train_data="${data_path}/phase0"
max_steps0=30000
OUTFILE0="${RESULTS_DIR}/${EXP_NAME}/slurm-phase0-%j-%n.out"
ERRFILE0="${RESULTS_DIR}/${EXP_NAME}/error-phase0-%j-%n.out"

phase1_dir=${result_dir}/phase1
phase1_train_data="${data_path}/phase1"
max_steps1=50000
OUTFILE1="${RESULTS_DIR}/${EXP_NAME}/slurm-phase1-%j-%n.out"
ERRFILE1="${RESULTS_DIR}/${EXP_NAME}/error-phase1-%j-%n.out"

phase2_dir=${result_dir}/phase2
phase2_train_data="${data_path}/phase2"
max_steps2=75000
OUTFILE2="${RESULTS_DIR}/${EXP_NAME}/slurm-phase2-%j-%n.out"
ERRFILE2="${RESULTS_DIR}/${EXP_NAME}/error-phase2-%j-%n.out"

phase3_dir=${result_dir}/phase3
phase3_train_data="${data_path}/phase3"
max_steps3=75000
OUTFILE3="${RESULTS_DIR}/${EXP_NAME}/slurm-phase3-%j-%n.out"
ERRFILE3="${RESULTS_DIR}/${EXP_NAME}/error-phase3-%j-%n.out"

phase4_dir=${result_dir}/phase4
phase4_train_data="${data_path}/phase4"
max_steps4=100000
OUTFILE4="${RESULTS_DIR}/${EXP_NAME}/slurm-phase4-%j-%n.out"
ERRFILE4="${RESULTS_DIR}/${EXP_NAME}/error-phase4-%j-%n.out"

# Diagnostic prints
echo "Hosts: $hosts"
# Command to be run


if [ ! -f ${phase0_dir}/best.ckpt ]; then
  if compgen -G ${phase0_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
    resume0=true
  else
    resume0=false
  fi \
  && read -r -d '' cmd0 <<EOF
echo "*******STARTING 0TH PHASE********" \
&& echo "---------------" \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${num_nodes} \
  trainer.gpus=${ngpus} \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase_train_data}/translations.txt \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  exp_manager.exp_dir=${phase0_dir} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps} \
  +exp_manager.resume_if_exists=${resume} \
  model.optim.lr=${phase0_lr} \
&& echo  "*******FINISHING 0TH FHASE********"
EOF \
  && srun -o $OUTFILE0 -e $ERRFILE0 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd0}"
fi \
&& if [ ! -f ${phase1_dir}/best.ckpt ]; then
  && if compgen -G ${phase1_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
    resume1=true
  else
    resume1=false
  fi \
  && read -r -d '' cmd1 <<EOF
echo "*******STARTING 1ST PHASE********" \
&& echo "---------------" \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${num_nodes} \
  trainer.gpus=${ngpus} \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase_train_data1}/originals.txt \
  model.train_ds.tgt_file_name=${phase_train_data1}/translations.txt \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  exp_manager.exp_dir=${phase1_dir} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps1} \
  +exp_manager.resume_if_exists=${resume1} \
  +model.weights_checkpoint=${phase0_dir}/best.ckpt \
  model.optim.lr=${back_translation_phases_lr} \
fi \
&& echo  "*******FINISHING 1ST FHASE********"
EOF \
  && srun -o $OUTFILE1 -e $ERRFILE1 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd1}"
fi \
&& if [ ! -f ${phase2_dir}/best.ckpt ]; then
  && if compgen -G ${phase2_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
    resume2=true
  else
    resume2=false
  fi \
  && read -r -d '' cmd2 <<EOF
echo "*******STARTING 2ND PHASE********" \
&& echo "---------------" \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${num_nodes} \
  trainer.gpus=${ngpus} \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase2_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase2_train_data}/translations.txt \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  exp_manager.exp_dir=${phase2_dir} \
  +exp_manager.resume_if_exists=${resume2} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps2} \
  +model.weights_checkpoint=${phase1_dir}/best.ckpt \
  model.optim.lr=${back_translation_phases_lr} \
&& echo  "*******FINISHING 2ND FHASE********"
EOF \
  && srun -o $OUTFILE2 -e $ERRFILE2 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd2}"
fi \
&& if [ ! -f ${phase3_dir}/best.ckpt ]; then
  && if compgen -G ${phase3_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
    resume3=true
  else
    resume3=false
  fi \
  && read -r -d '' cmd3 <<EOF
echo "*******STARTING 3RD PHASE********" \
&& echo "---------------" \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${num_nodes} \
  trainer.gpus=${ngpus} \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase3_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase3_train_data}/translations.txt \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  exp_manager.exp_dir=${phase3_dir} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps3} \
  +exp_manager.resume_if_exists=${resume3} \
  +model.weights_checkpoint=${phase2_dir}/best.ckpt \
  model.optim.lr=${back_translation_phases_lr} \
&& echo  "*******FINISHING 2ND FHASE********"
EOF \
  && srun -o $OUTFILE3 -e $ERRFILE3 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd3}"
fi \
&& if [ ! -f ${phase4_dir}/best.ckpt ]; then
  && if compgen -G ${phase4_dir}/TransformerMT/*/checkpoints/* > /dev/null; then
    resume4=true
  else
    resume4=false
  fi \
  && read -r -d '' cmd4 <<EOF
echo "*******STARTING 4TH PHASE********" \
&& echo "---------------" \
&& source /workspace/NeMo/examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_phases_post_traning/exports_en_de.sh \
&& python3 train.py -cn ${base_conf} \
  trainer.num_nodes=${num_nodes} \
  trainer.gpus=${ngpus} \
  model.train_ds.tokens_in_batch=${train_n_tokens_in_batch} \
  model.train_ds.src_file_name=${phase4_train_data}/originals.txt \
  model.train_ds.tgt_file_name=${phase4_train_data}/translations.txt \
  model.machine_translation.tokenizer.tokenizer_model=${tok_model} \
  model.validation_ds.src_file_name=${valid_src} \
  model.validation_ds.tgt_file_name=${valid_ref} \
  model.test_ds.src_file_name=${test_src} \
  model.test_ds.tgt_file_name=${test_ref} \
  exp_manager.exp_dir=${phase4_dir} \
  trainer.max_epochs=100000 \
  +trainer.max_steps=${max_steps4} \
  +exp_manager.resume_if_exists=${resume4} \
  +model.weights_checkpoint=${phase3_dir}/best.ckpt \
  model.optim.lr=${back_translation_phases_lr} \
&& echo  "*******FINISHING 4TH FHASE********"
EOF \
  && srun -o $OUTFILE4 -e $ERRFILE4 \
    --container-image="$CONTAINER" \
    $MOUNTS \
    bash -c "${cmd4}"
fi \

set +