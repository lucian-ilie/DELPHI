#!/bin/bash
compute_HYD="python3 ${PRO_DIR}/feature_computation/HYD/compute.py"
${compute_HYD} ${INPUT_FN} ${TMP_DIR}/HYD.txt