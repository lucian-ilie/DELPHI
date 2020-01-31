#!/bin/bash
set -x

split_program="/home/j00492398/test_joey/interface-pred/tools/split_fasta_file_by_protein/split.sh"
RSA_compute_program="python /home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/RSA/compute.py"
# $1: input fasta fn
# $2: DS_prefix
compute_RSA () {
    DS_prefix=$2
    output_RSA_fn=/home/j00492398/test_joey/interface-pred/Features/RSA/$2.txt
    raw_rsa_dir=/home/j00492398/test_joey/raw_features/putativeRSA/${DS_prefix}
    OneProPerFileDIR=/home/j00492398/test_joey/raw_features/one_file_per_protein/${DS_prefix}/
#    mkdir -p ${raw_rsa_dir}
#    # shellcheck disable=SC2115
#    rm -r ${raw_rsa_dir}/*
#    mkdir -p ${OneProPerFileDIR}
#    # shellcheck disable=SC2115
#    rm -r ${OneProPerFileDIR}*
#    ${split_program} "$1" ${OneProPerFileDIR}
#
#    #step two use ASA to compute raw_rsa
#    cd ${raw_rsa_dir}
#    for filename in ${OneProPerFileDIR}*
#    do
#        ASAquick $filename
#    done

    #output one file
    ${RSA_compute_program}  $1 ${raw_rsa_dir}  ${output_RSA_fn}

}

DS_prefix="DS_72"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_72/DS_72_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="DS_164"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="DS_186"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="SCRIBER_test"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="SCRIBER_train"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_train_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="survey_test"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_test_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}

DS_prefix="survey_train"
ProFastaFN="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_train_Pid_and_Pseq.txt"
compute_RSA ${ProFastaFN} ${DS_prefix}


