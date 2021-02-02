#!/bin/bash

# fasttext model has to be in wikimatrix directory.
for d in ../*; do
  echo "working on $d"
  python ~/PeganovNeMo/utils/filter_by_language.py -s $d/en \
    -t $d/en \
    -l en \
    -r $d/en_garbage \
    -S $d/de \
    -T $d/de \
    -L de \
    -R $d/de_garbage \
    --fasttext-model lid.176.bin
done