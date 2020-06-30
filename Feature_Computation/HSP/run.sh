#!/bin/bash
compute="python3 /home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/HSP/compute.py"
train_fn="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_train_nonRed_Pid_and_Pseq_label.txt"
#target_fn="/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/HSP/test_target.txt"
hsp="/home/j00492398/test_joey/interface-pred/Dataset/72_164_186_SCRIBER_test_survey_train_Pid_and_Pseq.HSP"
#out_fn="/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/HSP/test_out.txt"
#${compute} ${train_fn} ${target_fn} ${hsp} ${out_fn}
seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_72/DS_72_Pid_and_Pseq.txt"
out_fn_prefix="DS_72"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_and_Pseq.txt"
out_fn_prefix="DS_164"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_and_Pseq.txt"
out_fn_prefix="DS_186"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_train_Pid_and_Pseq.txt"
out_fn_prefix="survey_train"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_Pid_and_Pseq.txt"
out_fn_prefix="SCRIBER_test"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_train_Pid_and_Pseq.txt"
out_fn_prefix="SCRIBER_train"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt

seq="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_test_Pid_and_Pseq.txt"
out_fn_prefix="survey_test"
${compute} ${train_fn} ${seq} ${hsp} /home/j00492398/test_joey/interface-pred/Features/HSP/${out_fn_prefix}.txt