set -x
moses_path=~/mosesdecoder
input=$1
start_i=$2
end_i=$3
lang=$4
mkdir -p tmp
sed -n "${start_i},${end_i}p" ${input} | \
  perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l ${lang} | \
  perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > \
  tmp/rank$5
set +x