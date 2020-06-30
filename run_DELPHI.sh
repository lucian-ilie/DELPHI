#!/bin/bash
#set -x
# This is the DELPHI program entrance
# Usage: ./run_DELPHI.sh [INPUT_FN]
which python
export INPUT_FN=$1
# OUT_DIR=${PRO_DIR}/tmp_human_pssm_last1000
# export TMP_DIR=${PRO_DIR}/tmp-$(date +%Y-%m-%d-%H-%M-%S)
# export TMP_DIR=${PRO_DIR}/tmp_human_pssm_last1000
echo "PRO_DIR: $PRO_DIR"
echo "TMP_DIR: $TMP_DIR"

#####################
#check PSSM database#
#####################
# argv[2]: PSSM_DIR=${TMP_DIR}/PSSM_raw/1/
echo "load_PSSM_DB"
python3 utils/load_PSSM_DB.py ${INPUT_FN} ${TMP_DIR}/PSSM_raw/1

####################
# compute features#
####################
bash feature_computation/compute_features.sh $INPUT_FN

# ####################
# #    run DELPHI    #
# ####################
python3 predict.py -i $INPUT_FN -d $TMP_DIR -o $OUT_DIR -c 1
if [ $? -ne 0 ]
then
   echo "[Error:] DELPHI returns 1!"
fi