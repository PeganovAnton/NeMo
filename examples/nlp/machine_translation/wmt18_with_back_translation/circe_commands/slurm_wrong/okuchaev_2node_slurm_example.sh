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
BATCH_SIZE=32
EPOCHS=2
lr=0.005
# Env Variable Setup
set -x
CONTAINER="gitlab-master.nvidia.com/okuchaiev/nemo_containers:nemo-1.0.0a0"
# Directories for manifests, data, etc.
DATA_DIR='/gpfs/fs1/scratch/jocelynh/nemo_asr/ASR_SET_1.2'
TRAIN_MANIFEST='/data/train/tarred_train_all.json'
LS_DATA='/gpfs/fs1/datasets/jasper-joc/LibriSpeech'
RESULTS_DIR='/gpfs/fs1/okuchaiev/results/newnemo'
CODE_DIR='/gpfs/fs1/okuchaiev/repos/NeMo'
MANIFESTS='/gpfs/fs1/jbalam/manifests'
EXP_NAME='tst2node'
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo
mkdir -p $RESULTS_DIR/$EXP_NAME
MOUNTS="--container-mounts=$DATA_DIR:/data,$CODE_DIR:/code,$RESULTS_DIR/$EXP_NAME:/results,$LS_DATA:/data/LibriSpeech,$MANIFESTS:/manifests"
# Set up multinode if applicable
SRUN_HOST=''
SRUN_CONT=''
if [[ -z "$SLURM_JOB_ID" ]]; then
  hosts=( `hostname` )
  export SLURM_NNODES=1
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
  SRUN_HOST='srun --mem=0 -N 1 -n 1 -w $hostn'
  SRUN_CONT='srun --mem=0 -N 1 -n 1 -w $hostn --container-image="$CONTAINER" $MOUNTS'
fi
MASTER_IP="$(getent hosts "${hosts[0]}" | cut -d ' ' -f1 | head -n1)"
PORT=$((4242 + RANDOM%1000))
DGXSOCKETCORES=24
DGXNSOCKET=2
#if [[ $SLURM_NNODES -gt 1 ]]; then
#  export MULTI_NODE="--nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID --master_addr=$MASTER_IP --master_port=$PORT"
#else
#  export MULTI_NODE=""
#fi
# Diagnostic prints
echo "Hosts: $hosts"
echo "Master IP: $MASTER_IP"
echo "Port: $PORT"
echo "Slurm num nodes: $SLURM_NNODES"
export SLURM_NNODES=2
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$PORT
export WORLD_SIZE=32
export HYDRA_FULL_ERROR=1
# Command to be run
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&&  python /code/examples/asr/speech_to_text.py \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.train_ds.is_tarred=True \
    model.validation_ds.manifest_filepath=/manifests/librivox-dev-clean_circe.json \
    model.train_ds.batch_size=$BATCH_SIZE \
    trainer.gpus=-1 \
    trainer.num_nodes=$SLURM_NNODES \
    trainer.max_epochs=$EPOCHS \
    model.optim.lr=$lr \
    exp_manager.exp_dir=/results
EOF
OUTFILE="${RESULTS_DIR}/${EXP_NAME}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/${EXP_NAME}/error-%j-%n.out"
srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"
set +