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
  orinals_fn="stage${i}/originals.txt"
  cat "${par_train_src}" > "${originals_fn}"
  if [ -n "${stage_mono_sources[i]}" ]; then
    cat "${stage_mono_sources[i]}" >> "${originals_fn}"
  fi
  translations_fn="stage${i}/translations.txt"
  cat "${par_train_ref}" > "${translations_fn}"
  if [ -n "${stage_mono_refs[i]}" ]; then
    cat "${stage_mono_refs[i]}" >> "${translations_fn}"
  fi
done