pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && yttm bpe --data /data/train.clean.en-de.shuffled.common --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && ls /data \
  && python train.py -cn wmt16/en_de_8gpu \
  && python test.py -cn wmt16/en_de_8gpu

