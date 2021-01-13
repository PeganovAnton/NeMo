#!/bin/bash
# The only argument is wandb token.


read -r -d '' cmd <<EOF
git clone https://github.com/PeganovAnton/NeMo.git \
&& cd NeMo \
&& git checkout nmtmodel \
&& pip3 install -r requirements/requirements.txt \
&& pip3 install -r requirements/requirements_nlp.txt \
&& pip3 install --upgrade wandb \
&& pip3 install webdataset \
&& wandb login $1 \
&& export nemo_path="\$(pwd)" \
&& export HYDRA_FULL_ERROR=1 \
&& echo "NeMo path: \${nemo_path}" \
&& export PYTHONPATH="\${nemo_path}" \
&& cd  "\${nemo_path}/examples/nlp/machine_translation" \
&& yttm bpe --data /data/train.clean.en-de.shuffled.common --model bpe_37k_en_de_yttm.model --vocab_size 37000 \
&& python train.py --config-name=aayn_big \
  trainer.gpus=8 \
  ~trainer.max_epochs \
  model.encoder_tokenizer.tokenizer_model=bpe_37k_en_de_yttm.model  \
  model.encoder_tokenizer.bpe_dropout=0.1 \
  model.decoder_tokenizer.tokenizer_model=bpe_37k_en_de_yttm.model  \
  model.decoder_tokenizer.bpe_dropout=0.1 \
  model.train_ds.src_file_name=/data/train.clean.en.shuffled \
  model.train_ds.tgt_file_name=/data/train.clean.de.shuffled \
  model.validation_ds.src_file_name=/data/wmt13-en-de.src \
  model.validation_ds.tgt_file_name=/data/wmt13-en-de.ref \
  model.test_ds.src_file_name=/data/wmt14-en-de.src \
  model.test_ds.tgt_file_name=/data/wmt14-en-de.ref \
  model.optim.sched.warmup_steps=20000 \
  ~model.optim.sched.warmup_ratio \
  model.optim.lr=0.0005  \
  +model.optim.weight_decay=0.0002 \
  exp_manager.wandb_logger_kwargs.name=1st_trial \
  exp_manager.wandb_logger_kwargs.project=nmt_aayn_en_de_big_v0 \
  exp_manager.exp_dir=/result \
&& wmt13_translated=/result/newstest2013_en_translated.txt \
&& python3 nmt_transformer_infer.py \
  --model /result/best.ckpt \
  --srctext /data/wmt13_en_de.src \
  --tgtout "\${wmt13_translated}" \
  --target_lang de \
&& cat "\${wmt13_translated}" | sacrebleu -t wmt13 -l en-de \
&& wmt14_translated=/result/newstest2014_en_translated.txt \
&& python3 nmt_transformer_infer.py \
  --model /result/best.ckpt \
  --srctext /data/wmt14_en_de.src \
  --tgtout "\${wmt14_translated}" \
  --target_lang de \
&& cat "\${wmt14_translated}" | sacrebleu -t wmt14 -l en-de
EOF

ngc batch run \
  --name big_en_de_attention_is_all_you_need_close_to_paper \
  --preempt RUNONCE \
  --image nvidia/pytorch:20.09-py3 \
  --ace nv-us-west-2 \
  --instance dgx1v.32g.8.norm \
  --commandline "${cmd}" \
  --result /result \
  --org nvidian \
  --team ac-aiapps \
  --datasetid 68792:/data

