#!/bin/bash
compute_POSITION="python3 ${PRO_DIR}/feature_computation/POSITION/compute.py"
${compute_POSITION} ${INPUT_FN} ${TMP_DIR}/POSITION.txt