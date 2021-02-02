#!/bin/bash

#Run in directory where dataset will be saved
set -x
mkdir raw
bash ~/PeganovNeMo/utils/download_wmt20_parallel.sh raw
find . -name "*.gz" -exec gzip -d {} \;
tar xzf commoncrawl/training-parallel-commoncrawl.tgz
rm commoncrawl/training-parallel-commoncrawl.tgz
mv commoncrawl.* commoncrawl/
cd commoncrawl/
rm commoncrawl.cs* commoncrawl.es* commoncrawl.de-en.annotation
cd ..
python ~/PeganovNeMo/utils/extract_data_from_xlf_format.py rapid/RAPID_2019.de-en.xlf
rm rapid/RAPID_2019.de-en.xlf
python ~/PeganovNeMo/utils/extract_parallel_from_tsv.py europarl/europarl-v10.de-en.tsv 0 europarl/de 1 europarl/en