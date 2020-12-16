#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=dl-langspeech
#SBATCH -o en_de_parallel_logs.txt


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

export image_and_mounts='--container-image nvcr.io#nvidia/pytorch:20.11-py3 '\
'--container-mounts "/home/apeganov/result_de_en_back_translation_4_stages_1st_trial:/workspace/result_de_en_back_translation_4_stages_1st_trial,'\
'/gpfs/fs1/apeganov/datasets:/workspace/mydatasets,'\
'/home/apeganov/PeganovNeMo:/workspace/PeganovNeMo"'

# run script from above
srun ${image_and_mounts} \
    bash -c 'cd PeganovNeMo && echo "Current directory after first command:" $(pwd)'
srun ${image_and_mounts} --nodes 1 \
    bash -c 'source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/slurm/install/standard.sh'
srun ${image_and_mounts} \
    bash -c 'source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/slurm/exports_en_de.sh 8 16'
srun ${image_and_mounts} \
    bash -c 'cd PeganovNeMo && source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_stages_post_traning/attention_is_all_you_need.sh 0'
