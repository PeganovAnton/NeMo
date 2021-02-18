#!/bin/bash
#SBATCH -A dl-langspeech
#SBATCH -p batch
#SBATCH -N 1                    # number of nodes
#SBATCH -t 8:00:00              # wall time
#SBATCH -J "translate"   # job name
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

# You have to provide following variables:
# DATA_DIR -- path to tar files
# WANDB_TOKEN
# MODEL -- the name of a model from rc1 folder
# TAR_TEMPLATE -- template of tar files (see tarred dataset documentation)
# SOURCE_LANG
# TARGET_LANG


WANDB_PROJECT=back_translation_de_en
CONTAINER="rumpelschtilzchen/pytorch-20.11-py3-nemo-nmt:latest"
# Pass DATA_DIR as a parameter
data_dir=/data
RESULT_DIR='/home/apeganov'
result_dir=/result
PRETRAINED_MODELS='/gpfs/fs1/apeganov/models/mt/wmt_translate_models/rc1'
pretrained_models='/pretrained'
EXP_NAME='translate_de_en'
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
mkdir -p $RESULT_DIR/$EXP_NAME
MOUNTS="--container-mounts=$DATA_DIR:/data,$RESULT_DIR/$EXP_NAME:${result_dir}:${PRETRAINED_MODELS}:${pretrained_models}"

tar_files=${data_dir}/${TAR_TEMPLATE}
metadata=${data_dir}/metadata.json

OUTFILE="${RESULT_DIR}/${EXP_NAME}/slurm-%j-%n.out"
ERRFILE="${RESULT_DIR}/${EXP_NAME}/error-%j-%n.out"

read -r -d '' cmd <<EOF
if [ "\${SLURM_PROCID}" -eq "0" ]; then wandb login ${WANDB_TOKEN}; fi
nemo_path="\$(pwd)"
echo "NeMo path: \${nemo_path}"
export PYTHONPATH="\${nemo_path}"
cd "\${nemo_path}/examples/nlp/machine_translation"

python translate_ddp.py \
  --model=${pretrained_models}/${MODEL} \
  --text2translate=${tar_files} \
  --metadata_path ${metadata} \
  --result_dir ${result_dir} \
  --source_lang ${SOURCE_LANG} \
  --target_lang ${TARGET_LANG}
set +e +x
EOF

srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"

