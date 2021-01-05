pip3 install -r requirements/requirements.txt \
  && pip3 install -r requirements/requirements_nlp.txt \
  && pip3 install webdataset \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && yttm bpe --data /data/train.clean.en-de.shuffled.common --model bpe_37k_en_de_yttm.model --vocab_size 37000 \
  && python3 train.py -cn wmt16_big/en_de_8gpu \
    model.machine_translation.tokenizer.tokenizer_model=bpe_37k_en_de_yttm.model \
    model.machine_translation.embedding_dropout=0.3 \
    model.machine_translation.ffn_dropout=0.3 \
    model.machine_translation.attn_score_dropout=0.3 \
    model.machine_translation.attn_layer_dropout=0.3 \
    model.train_ds.tokens_in_batch=10000 \
    +trainer.max_steps=300000 \
    trainer.max_epochs=300000 \
    model.optim.sched.warmup_steps=10000 \
    model.optim.sched.warmup_ratio=null \
    model.optim.lr=0.0005 \
  && wmt13_translated=/result/newstest2013_en_translated.txt \
  && python3 nmt_transformer_infer.py \
    --model /result/best.ckpt \
    --text2translate /data/wmt13_en_de.src \
    --output ${wmt13_translated} \
  && cat ${wmt13_translated} | sacrebleu -t wmt13 -l en-de \
  && wmt14_translated=/result/newstest2014_en_translated.txt \
  && python3 nmt_transformer_infer.py \
    --model /result/best.ckpt \
    --text2translate /data/wmt14_en_de.src \
    --output ${wmt14_translated} \
  && cat ${wmt14_translated} | sacrebleu -t wmt14 -l en-de

