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

# Diagnostic prints
echo "Hosts: $hosts"
# Command to be run
read -r -d '' cmd <<EOF
echo "*******STARTING BACK TRANSLATION TRANING********" \
&& echo "---------------" \
&& source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_stages_post_traning/exports_en_de.sh \
&& source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_stages_post_traning/attention_is_all_you_need.sh \
&& echo  "*******FINISHING BACK TRANSLATION TRAINING********"
EOF

OUTFILE="${RESULTS_DIR}/${EXP_NAME}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/${EXP_NAME}/error-%j-%n.out"
srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"
set +