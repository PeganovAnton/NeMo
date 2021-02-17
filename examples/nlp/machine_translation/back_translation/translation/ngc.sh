#!/bin/bash


INSTANCE=dgx1v.32g.8.norm
PROJECT=z
DATAID=74384
DATA_PATH=/data
WORKSPACE=wmt_translate_models
WANDBLOGIN=$1
RAID_MONO=/raid/mono
RAID_TRANSLATED=/raid/translated
RESULTS_DIR=/results
WORKSPACE_POINT=/models
MODEL=$2
TAR_TEMPLATE=$3
METADATA=$4
SOURCE_LANG=$5
TARGET_LANG=$6

read -r -d '' cmd <<EOF
set -e -x
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
nvidia-smi
apt-get update
apt-get install -y libsndfile1 ffmpeg
pip install --upgrade wandb
pip install Cython
wandb login ${WANDBLOGIN}
git clone https://github.com/PeganovAnton/NeMo.git
cd NeMo
git checkout sacreBLEU_pl_metric
./reinstall.sh
mkdir -p ${RAID_MONO} $RAID_TRANSLATED}
cp -R ${DATA_PATH}/* ${RAID_MONO}
tree -hs ${DATA_PATH}
tree -hs ${RAID_MONO}
python examples/nlp/machine_translation/translate_ddp.py \
  --model=${WORKSPACE_POINT}/rc1/${MODEL} \
  --text2translate=${RAID_MONO}/${TAR_TEMPLATE} \
  --metadata_path ${RAID_MONO}/${METADATA} \
  --result_dir ${RAID_TRANSLATED} \
  --source_lang ${SOURCE_LANG} \
  --target_lang ${TARGET_LANG}
tar czf ${RESULTS_DIR}/translated.tar.gz ${RAID_TRANSLATED}
set +e +x
EOF
echo $cmd

ngc batch run --name "backtranslation_en_de_wmt20" --preempt RUNONCE \
    --image "nvcr.io/nvidia/pytorch:20.11-py3" \
    --ace nv-us-west-2 \
    --instance $INSTANCE \
    --commandline "${cmd}" \
    --result /results/ \
    --org nvidian \
    --team ac-aiapps \
    --datasetid $DATAID:${DATA_PATH}/ \
    --workspace $WORKSPACE:${WORKSPACE_POINT}:RO


