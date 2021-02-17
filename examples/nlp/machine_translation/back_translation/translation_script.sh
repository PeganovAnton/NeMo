#!/bin/bash


INSTANCE=dgx1v.16g.8.norm
PROJECT=back_translation_de_en
DATAID=74340
WORKSPACE=wmt_translate_models
WANDBLOGIN=$1
RAID_MONO=/raid/mono
RAID_TRANSLATED=/raid/translated
RESULTS_DIR=/results
WORKSPACE_POINT=/models
MODEL=$2
TAR_TEMPLATE=$3
METADATA=$4

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
cp -R /data/* ${RAID_MONO}
shuf --random-source=/raid/monolingual.news.dedup.clean.tok.de /raid/monolingual.news.dedup.clean.tok.de \
  | head -10000000 > /raid/monolingual.news.dedup.clean.tok.de.sample
python examples/nlp/machine_translation/translate_ddp.py \
  --model=${WORKSPACE_POINT}/rc1/${MODEL} \
  --text2translate=${RAID_MONO}/${TAR_TEMPLATE} \
  --metadata_path ${METADATA} \
  --result_dir ${RAID_TRANSLATED}
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
    --datasetid $DATAID:/data/ \
    --workspace $WORKSPACE:${WORKSPACE_POINT}:RO


