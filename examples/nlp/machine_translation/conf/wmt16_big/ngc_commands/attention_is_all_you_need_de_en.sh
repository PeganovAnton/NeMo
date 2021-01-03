pip3 install -r requirements/requirements.txt \
  && pip3 install -r requirements/requirements_nlp.txt \
  && pip3 install webdataset \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && yttm bpe --data /data/train.clean.en-de.shuffled.common --model bpe_16k_en_de_yttm.model --vocab_size 16000 \
  && python train.py -cn wmt16_big/de_en_8gpu \
  && python test.py -cn wmt16_big/de_en_8gpu