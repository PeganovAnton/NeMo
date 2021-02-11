moses_path=~/mosesdecoder
mkdir tmp
sed -n "$3,$4p" $1 | \
  perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $5 | \
  perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > \
  tmp/rank$6
cat tmp/rank* > $2
rm -r tmp
