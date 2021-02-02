urls=(
  "http://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz"
  "https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-de.txt.gz"
  "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
  "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.de-en.tsv.gz"
  "http://data.statmt.org/wikititles/v2/wikititles-v2.de-en.tsv.gz"
  "http://data.statmt.org/wmt20/translation-task/rapid/RAPID_2019.de-en.xlf.gz"
  "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.de-en.langid.tsv.gz"
)

DP=$1

locs=(
  "${DP}/europarl/europarl-v10.de-en.tsv.gz"
  "${DP}/paracrawl/en-de.txt.gz"
  "${DP}/commoncrawl/training-parallel-commoncrawl.tgz"
  "${DP}/news-commentary/news-commentary-v15.de-en.tsv.gz"
  "${DP}/wikititles/wikititles-v2.de-en.tsv.gz"
  "${DP}/rapid/RAPID_2019.de-en.xlf.gz"
  "${DP}/wikimatrix/WikiMatrix.v1.de-en.langid.tsv.gz"
)

for f in ${locs[@]}; do mkdir -p "$(dirname ${f})"; done

for i in ${!urls[@]}; do echo "${urls[i]}" "${locs[i]}"; done | \
  xargs -n 2 -P "${#locs[@]}" sh -c 'wget -q "$1" -O "$2"' sh