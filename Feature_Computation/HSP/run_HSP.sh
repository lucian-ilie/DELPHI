#!/bin/bash
SPRINT_program=${PRO_DIR}/../programs/SPRINT/bin/compute_HSPs
SPRINT_DB=${PRO_DIR}/feature_computation/HSP/SPRING_DB_14836_Pid_and_Pseq.txt
SPRINT_train_pid_pseq_label=${PRO_DIR}/feature_computation/HSP/SPRING_DB_14836_Pid_and_Pseq_label.txt
compute="python3 ${PRO_DIR}/feature_computation/HSP/compute.py"
mkdir ${TMP_DIR}/HSP_raw
if [ ! -f "${TMP_DIR}/HSP_raw/raw.hsp" ]; then
${SPRINT_program} -p ${SPRINT_DB} -h ignore_hsp.txt -add ${INPUT_FN} ${TMP_DIR}/HSP_raw/raw.hsp
else
	echo "${TMP_DIR}/HSP_raw/raw.hsp already exist"
fi
${compute} ${SPRINT_train_pid_pseq_label} ${INPUT_FN} ${TMP_DIR}/HSP_raw/raw.hsp ${TMP_DIR}/HSP.txt