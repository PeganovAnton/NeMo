pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd /data \
  && cat train.en train.de > yttm_train.ende \
  && echo "current path when creating yttm model: $(pwd)" \
  && yttm bpe --data yttm_train.ende --model bpe_37k_en_de_yttm.model --vocab_size 37000 \
  && mkdir -p wmt14_en_de2 \
  && cd wmt14_en_de2 \
  && cp ../bpe_37k_en_de_yttm.model ./ \
  && cp ../test* ./ \
  && cp ../valid* ./ \
  && cp valid.en train.en \
  && cp valid.de train.de \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && python train.py -cn debug_on_ngc \
  && python test.py model.test_checkpoint_path=best.ckpt -cn debug_on_ngc

