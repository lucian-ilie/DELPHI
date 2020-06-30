#!/bin/bash
# set -x
echo "INPUT_FN is $INPUT_FN"
split_program="${PRO_DIR}/utils/split.sh"
genMSA_program="${PRO_DIR}/feature_computation/ECO/genMSA.sh"
cleanPSSM_program="${PRO_DIR}/feature_computation/ECO/cleanPSSM.sh"
compute_ECO_program="python3 ${PRO_DIR}/feature_computation/ECO/readFileGeneric.py"

output_ECO_fn=${TMP_DIR}/ECO.txt
raw_eco_dir1=${TMP_DIR}/ECO_raw/1/
raw_eco_dir2=${TMP_DIR}/ECO_raw/2/
OneProPerFileDIR=${TMP_DIR}/ECO_raw/3/
mkdir -p ${raw_eco_dir1}
mkdir -p ${raw_eco_dir2}
mkdir -p ${OneProPerFileDIR}
# # shellcheck disable=SC2115
# rm -r ${raw_eco_dir1}/*
# # shellcheck disable=SC2115
# rm -r ${raw_eco_dir2}/*
# # shellcheck disable=SC2115
# rm -r ${OneProPerFileDIR}/*
echo "split_program"
${split_program} ${INPUT_FN} ${OneProPerFileDIR}
echo "genMSA_program"
${genMSA_program} ${OneProPerFileDIR} ${raw_eco_dir1}
echo "cleanPSSM_program"
${cleanPSSM_program} ${raw_eco_dir1} ${raw_eco_dir2}
echo "compute_ECO_program"
${compute_ECO_program} ${OneProPerFileDIR} ${raw_eco_dir2} ${output_ECO_fn} $INPUT_FN
cd ${PRO_DIR}