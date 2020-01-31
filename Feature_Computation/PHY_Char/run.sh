#!/bin/bash
compute="python /home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/PHY_Char/compute.py"

seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_72/DS_72_Pid_and_Pseq.txt"
out_fn_prefix="DS_72"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_and_Pseq.txt"
out_fn_prefix="DS_164"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_and_Pseq.txt"
out_fn_prefix="DS_186"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_train_Pid_and_Pseq.txt"
out_fn_prefix="survey_train"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_Pid_and_Pseq.txt"
out_fn_prefix="SCRIBER_test"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_train_Pid_and_Pseq.txt"
out_fn_prefix="SCRIBER_train"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}

seq="/home/j00492398/test_joey/interface-pred/Dataset/survey/survey_test_Pid_and_Pseq.txt"
out_fn_prefix="survey_test"
${compute} ${seq} /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${out_fn_prefix}