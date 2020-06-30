#!/bin/bash
#set -x

split_program="${PRO_DIR}/utils/split.sh"
Anchor_compute_program="python3 ${PRO_DIR}/feature_computation/Anchor/load.py"

output_Anchor_fn=${TMP_DIR}/Anchor.txt
raw_Anchor_dir=${TMP_DIR}/Anchor_raw/1/
OneProPerFileDIR=${TMP_DIR}/Anchor_raw/2/
mkdir -p ${raw_Anchor_dir}
mkdir -p ${OneProPerFileDIR}
${split_program} ${INPUT_FN} ${OneProPerFileDIR}

#step two use Anchor to compute raw_anchor
cd ${raw_Anchor_dir}
for filename in ${OneProPerFileDIR}*
do
    f_basename=$(basename -- "$filename")
    anchor $filename -d ${PRO_DIR}/../programs/ANCHOR/ > ${raw_Anchor_dir}/${f_basename}.anchor
done

cd ${PRO_DIR}
${Anchor_compute_program} ${INPUT_FN} ${raw_Anchor_dir} ${output_Anchor_fn}

