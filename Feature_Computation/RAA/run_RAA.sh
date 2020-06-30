#!/bin/bash
compute_RAA="python3 ${PRO_DIR}/feature_computation/RAA/compute.py"
${compute_RAA} ${INPUT_FN} ${TMP_DIR}/RAA.txt
