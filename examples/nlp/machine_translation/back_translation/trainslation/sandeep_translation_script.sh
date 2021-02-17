#!/bin/bash
INSTANCE=dgx1v.32g.8.norm
PROJECT=backtranslation-en-de-wmt20
DATAID=70513
WORKSPACE=wmt_translate_models
WANDBLOGIN=<wandb>
set -e
ngc batch run --name "backtranslation_en_de_wmt20" --preempt RUNONCE \
    --image "nvcr.io/nvidia/pytorch:20.09-py3" \
    --ace nv-us-west-2 \
    --instance $INSTANCE \
    --commandline "export GLOO_SOCKET_IFNAME=eth0 && export NCCL_SOCKET_IFNAME=eth0 && nvidia-smi && apt-get update && apt-get install -y libsndfile1 ffmpeg && \
    pip install --upgrade wandb && pip install Cython && wandb login a380d6dd9912914b6b85dbd396f528c5fc465766 && \
    git clone https://github.com/NVIDIA/NeMo.git && cd NeMo && \
    git checkout nmt_dataset_caching && ./reinstall.sh && \
    cp -R /data/mono/monolingual.news.dedup.clean.tok.de /raid/ && \
    shuf --random-source=/raid/monolingual.news.dedup.clean.tok.de /raid/monolingual.news.dedup.clean.tok.de | head -10000000 > /raid/monolingual.news.dedup.clean.tok.de.sample && \
    python examples/nlp/machine_translation/translate_ddp.py \
        --model=/models/wmt20_de_en_big/AAYNBase/2021-01-01_23-22-33/checkpoints/AAYNBase.nemo \
        --text2translate=/raid/monolingual.news.dedup.clean.tok.de.sample \
        --max_num_tokens_in_batch 8000 \
        --result_dir /results" \
    --result /results/ \
    --org nvidian \
    --team swdl-ai-apps \
    --datasetid $DATAID:/data/ \
    --workspace $WORKSPACE:/models/