#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=dl-langspeech


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun --container-image nvcr.io#nvidia/pytorch:20.09-py3 \
  --container-mounts "/home/apeganov/result_de_en_back_translation_4_stages_1st_trial:/workspace/result_de_en_back_translation_4_stages_1st_trial,/home/apeganov/old_results:/workspace/old_results,/gpfs/fs1/apeganov/datasets:/workspace/mydatasets/apeganov,/home/apeganov/PeganovNeMo:/workspace/PeganovNeMo" \
  source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/slurm/submit_en_de.slurm 8 16
