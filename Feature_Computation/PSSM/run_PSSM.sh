#!/bin/bash
# set -x
split_program="${PRO_DIR}/utils/split.sh"
OneProPerFileDIR=${TMP_DIR}/PSSM_raw/2/
PSSM_DIR=${TMP_DIR}/PSSM_raw/1/
mkdir -p ${OneProPerFileDIR}
mkdir -p ${PSSM_DIR}
${split_program} ${INPUT_FN} ${OneProPerFileDIR}
    
for filename in $(ls ${OneProPerFileDIR}*)
# for filename in ${OneProPerFileDIR}*
do
	f_basename=$(basename -- "$filename")
	pssm_f=${PSSM_DIR}${f_basename}.pssm
	if [ ! -f "$pssm_f" ]; then
	echo "${pssm_f} doesn't exist, compute it"
    # cat ${filename} >> /project/ctb-ilie/yli922/pssm/dataset/survey_train_Pid_and_Pseq_2kleft.txt
    psiblast -query $filename -db ${PRO_DIR}/../blastDB/nr -num_threads 5 -out_ascii_pssm ${pssm_f} -num_iterations 3
	else
		echo "${pssm_f} already exist"
	fi
done

compute="python3 ${PRO_DIR}/feature_computation/PSSM/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/PSSM ${PSSM_DIR}