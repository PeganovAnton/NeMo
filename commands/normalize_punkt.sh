set -x
moses_path=~/mosesdecoder
mkdir -p tmp
sed -n "$3,$4p" $1 | \
  perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $5 | \
  perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > \
  tmp/rank$6
set +x