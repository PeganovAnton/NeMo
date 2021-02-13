set -x
moses_path=~/mosesdecoder
mkdir -p tmp
sed -n "$2,$3p" $1 | \
  perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $4 | \
  perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > \
  tmp/rank$5
set +x