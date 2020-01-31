#!/bin/bash
set -x

split_program="/home/j00492398/test_joey/interface-pred/tools/split_fasta_file_by_protein/split.sh"
genMSA_program="/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/ECO/genMSA.sh"
cleanPSSM_program="/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/ECO/cleanPSSM.sh"
compute_ECO_program="python /home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/ECO/readFileGeneric.py"
# $1: input fasta fn
# $2: DS_prefix
compute_ECO() {
  DS_prefix=$2
  output_ECO_fn=/home/j00492398/test_joey/interface-pred/Features/ECO/$2.txt
  raw_eco_dir1=/home/j00492398/test_joey/raw_features/ECO/${DS_prefix}/1/
  raw_eco_dir2=/home/j00492398/test_joey/raw_features/ECO/${DS_prefix}/2/
  OneProPerFileDIR=/home/j00492398/test_joey/raw_features/one_file_per_protein/${DS_prefix}/
  mkdir -p ${raw_eco_dir1}
  mkdir -p ${raw_eco_dir2}
  mkdir -p ${OneProPerFileDIR}
  # shellcheck disable=SC2115
  rm -r ${raw_eco_dir1}/*
  # shellcheck disable=SC2115
  rm -r ${raw_eco_dir2}/*
  # shellcheck disable=SC2115
  rm -r ${OneProPerFileDIR}*
  ${split_program} "$1" ${OneProPerFileDIR}

  # step generate MSA
  ${genMSA_program} ${OneProPerFileDIR} ${raw_eco_dir1}
  ${cleanPSSM_program} ${raw_eco_dir1} ${raw_eco_dir2}
  # output one file
  ${compute_ECO_program} ${OneProPerFileDIR} ${raw_eco_dir2} ${output_ECO_fn} $1
}

#DS_prefix="DS_72"
#ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_72/DS_72_Pid_and_Pseq.txt"
#compute_ECO ${ProFastaFN} ${DS_prefix}

DS_prefix="DS_164"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_and_Pseq.txt"
compute_ECO ${ProFastaFN} ${DS_prefix}

DS_prefix="DS_186"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_and_Pseq.txt"
compute_ECO ${ProFastaFN} ${DS_prefix}
