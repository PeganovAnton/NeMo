#!/bin/bash

#Run in directory where dataset will be saved
set -x -e
SCRIPTS=~/PeganovNeMo/utils
moses_path=~/mosesdecoder
if [ -d raw ]; then
  rm -r raw
fi
mkdir raw
source ~/PeganovNeMo/commands/download_wmt20_parallel.sh raw
cp -r raw archive
cd raw
find . -name "*.gz" -exec gzip -d {} \;
tar xzf commoncrawl/training-parallel-commoncrawl.tgz
rm commoncrawl/training-parallel-commoncrawl.tgz
mv commoncrawl.* commoncrawl/
cd commoncrawl/
rm commoncrawl.cs* commoncrawl.es* commoncrawl.de-en.annotation commoncrawl.fr* commoncrawl.ru*
mv commoncrawl.de-en.de de
mv commoncrawl.de-en.en en
cd ..
python ${SCRIPTS}/extract_data_from_xlf_format.py rapid/RAPID_2019.de-en.xlf
rm rapid/RAPID_2019.de-en.xlf
python ${SCRIPTS}/balance_number_of_delimeters.py europarl/europarl-v10.de-en.tsv europarl/europarl-v10.de-en.tsv $'\t'
python ${SCRIPTS}/extract_parallel_from_tsv.py europarl/europarl-v10.de-en.tsv 0 europarl/de 1 europarl/en
python ${SCRIPTS}/extract_parallel_from_tsv.py news-commentary/news-commentary-v15.de-en.tsv \
    0 news-commentary/de 1 news-commentary/en
python ${SCRIPTS}/extract_parallel_from_tsv.py paracrawl/en-de.txt 0 paracrawl/en 1 paracrawl/de
python ${SCRIPTS}/extract_parallel_from_tsv.py wikimatrix/WikiMatrix.v1.de-en.langid.tsv 1 wikimatrix/de 2 wikimatrix/en
python ${SCRIPTS}/extract_parallel_from_tsv.py wikititles/wikititles-v2.de-en.tsv 0 wikititles/de 1 wikititles/en
rm -r */*.tsv

for d in *; do
  python ~/PeganovNeMo/utils/filter_by_language.py -s $d/en \
    -t $d/de \
    -S $d/en \
    -T $d/de \
    -l en \
    -L de \
    -r $d/garbage.en \
    -R $d/garbage.de \
    -m ../../lid.176.bin
done
cd wikimatrix
python ~/PeganovNeMo/utils/fix_quoting_in_wikimatrix.py -s en \
  -t de \
  -S en \
  -T de \
  -l en \
  -L de \
  --src-before en_before \
  --src-after en_after \
  --tgt-before de_before \
  --tgt-after de_after
cd ../..
mkdir -p cat_shuffled normalized
good_data=(commoncrawl europarl news-commentary rapid wikimatrix)
good_sources=($(for d in ${good_data[@]}; do echo raw/${d}/en; done))
good_targets=($(for d in ${good_data[@]}; do echo raw/${d}/de; done))
python ~/PeganovNeMo/utils/cat_dedup_shuffle.py -s ${good_sources[@]} -t ${good_targets[@]} -o cat_shuffled/en -r cat_shuffled/de

num_lines=$(wc -l < cat_shuffled/en)
num_cores=$(grep -c ^processor /proc/cpuinfo)
num_lines_per_core=$((${num_lines} / ${num_cores}))

start_i=($(seq 0 ${num_lines_per_core} $((${num_lines} - ${num_lines_per_core}))))
end_i=($(seq ${num_lines_per_core} ${num_lines_per_core} $((${num_lines} - ${num_lines_per_core}))) ${num_lines})
for i in ${!start_i[@]}; do echo ${start_i[i]} ${end_i[i]} $i; done | \
  xargs -n 3 -P "${num_cores}" \
    sh -c 'bash ~/PeganovNeMo/commands/normalize_punkt.sh cat_shuffled/en "$1" "$2" en "$3"' sh
cat tmp/rank* > normalized/en
rm -r tmp
for i in ${!start_i[@]}; do echo ${start_i[i]} ${end_i[i]} $i; done | \
  xargs -n 3 -P "${num_cores}" \
    sh -c 'bash ~/PeganovNeMo/commands/normalize_punkt.sh cat_shuffled/de "$1" "$2" de "$3"' sh
cat tmp/rank* > normalized/de $2
rm -r tmp
python ~/PeganovNeMo/utils/filter_alphabetically.py -s normalized/en \
  -t normalized/de \
  -S alphabetically_filtered/en \
  -T alphabetically_filtered/de \
  -r alphabetically_filtered/garbage.en \
  -R alphabetically_filtered/garbage.de \
  -l en \
  -L de \
  -f 0.5
$moses_path/scripts/training/clean-corpus-n.perl -ratio 1.3 final/train en de finalfinal/train 1 250
set +x +e