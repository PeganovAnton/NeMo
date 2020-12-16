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

export image='nvcr.io#nvidia/pytorch:20.11-py3'
export mounts='/home/apeganov/result_de_en_back_translation_4_stages_1st_trial:/workspace/result_de_en_back_translation_4_stages_1st_trial,/gpfs/fs1/apeganov/datasets:/workspace/mydatasets,/home/apeganov/PeganovNeMo:/workspace/PeganovNeMo'

export container_save_path=saved_4_stages_container

# run script from above
srun --container-image=${image} --container-mounts=${mounts} --container-save ${container_save_path} \
    bash -c 'cd PeganovNeMo && echo "Current directory after first command:" $(pwd)'
srun --container-image=${container_save_path} --container-mounts=${mounts} --container-save ${container_save_path} --nodes 1 \
    bash -c 'cd PeganovNeMo && source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/slurm/install/standard.sh'
srun --container-image=${container_save_path} --container-mounts=${mounts} --container-save ${container_save_path} \
    bash -c 'cd PeganovNeMo && source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/slurm/exports_en_de.sh 8 16'
srun --container-image=${container_save_path} --container-mounts=${mounts} --container-save ${container_save_path} \
    bash -c 'cd PeganovNeMo && source examples/nlp/machine_translation/wmt18_with_back_translation/circe_commands/4_stages_post_traning/attention_is_all_you_need.sh 0'
