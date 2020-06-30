#!/bin/bash
compute_PKA="python3 ${PRO_DIR}/feature_computation/PKA/compute.py"
${compute_PKA} ${INPUT_FN} ${TMP_DIR}/PKA.txt