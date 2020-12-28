datasets_path=/gpfs/fs1/apeganov/datasets
wmt18_path=${datasets_path}/wmt18
four_stages_path=${wmt18_path}/4_stages_back_translation_en_de_path
nemo_path=~/PeganovNeMo
bi_path=${wmt18_path}/parallel_clean
par_train_src=${bi_path}/train.clean.filter.en.shuffled
par_train_ref=${bi_path}/train.clean.filter.de.shuffled
mono_wmt18_path=${wmt18_path}/de_mono_partially_translated_06.12.20_sampled
stage_mono_sources=( "" )
stage_mono_refs=( "" )
for md in 3M 6M 12M 24M
do
  stage_mono_sources+=( "${mono_data_path}/${md}/translations.txt" )
  stage_mono_refs+=( "${mono_data_path}/${md}/originals.txt" )
done
for i in 0 1 2 3 4
do
  echo "Composing phase ${i} train"
  stage_dir="stage${i}"
  mkdir -p "${stage_dir}"
  originals_fn="${stage_dir}/originals.txt"
  echo "Creating file ${originals_fn}"
  echo "Appending parallel data from ${par_train_src}"
  cat "${par_train_src}" > "${originals_fn}"
  if [ -n "${stage_mono_sources[i]}" ]; then
    echo "Appending mono data from ${stage_mono_sources[i]}"
    cat "${stage_mono_sources[i]}" >> "${originals_fn}"
  fi
  echo "Shuffling file ${originals_fn}"
  python3 ~/PeganovNeMo/utils/shuffle_pairs.py "${originals_fn}"
  translations_fn="${stage_dir}/translations.txt"
  echo "Creating file ${translations_fn}"
  echo "Appending parallel data from ${par_train_ref}"
  cat "${par_train_ref}" > "${translations_fn}"
  if [ -n "${stage_mono_refs[i]}" ]; then
    echo "Appending mono lingual data from ${stage_mono_refs[i]}"
    cat "${stage_mono_refs[i]}" >> "${translations_fn}"
  fi
  echo "Shuffling file ${translations_fn}"
  python3 ~/PeganovNeMo/utils/shuffle_pairs.py "${translations_fn}"
done