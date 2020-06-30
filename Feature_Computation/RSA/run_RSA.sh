#!/bin/bash
# set -x

split_program="${PRO_DIR}/utils/split.sh"
RSA_compute_program="python3 feature_computation/RSA/compute.py"
output_RSA_fn=${TMP_DIR}/RSA.txt
raw_rsa_dir=${TMP_DIR}/RSA_raw/1/
OneProPerFileDIR=${TMP_DIR}/RSA_raw/2/
mkdir -p ${raw_rsa_dir}
mkdir -p ${OneProPerFileDIR}
${split_program} ${INPUT_FN} ${OneProPerFileDIR}

#step two use ASA to compute raw_rsa
cd ${raw_rsa_dir}
for filename in ${OneProPerFileDIR}/*.fasta;
do
    ASAquick $filename
done

cd ${PRO_DIR}
#output one file
${RSA_compute_program} ${INPUT_FN} ${raw_rsa_dir} ${output_RSA_fn}

