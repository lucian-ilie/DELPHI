#!/bin/bash
#set -x
PRO_DIR="/home/j00492398/test_joey/interface-pred/"
program="python ./Load_and_Predict.py"
pro2vec_dic_fn="${PRO_DIR}Features/protVec_100d_3grams.csv"
training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_Pid_and_Pseq_label.txt"
#training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_7426_Pid_and_Pseq_label.txt"
train_db_prefix="survey_train"

Predict() {
testing_fn=$1
test_db_prefix=$2
gpu=1

for pa in {4,}; do
for nf in {-1,}; do
  for lr in {0.001,}; do
    for ms in {11,}; do
      for batch_size in {1024,}; do
        for lstm_unit in {32,}; do
          for filter_size in {32,}; do
            for kernel_size in {5,}; do
          for win in {31,}; do
            for min in {50,}; do
              for ep in {250,}; do
                for drop in {0.3,}; do
                  prefix="full_pa_${pa}_drop_${drop}_ms${ms}_lr${lr}_win${win}_min${min}_bs${batch_size}_nf${nf}_lstm${lstm_unit}_fil${filter_size}_ks${kernel_size}"
                  #prefix="debug"
                  mkdir -p logs/${prefix}
#                  mp="savedModel/${prefix}/SavedModelAndWeights.h5"
                  mp="/home/j00492398/test_joey/interface-pred/Src/csv/bestResults/17/SavedModelAndWeights.h5"
                  log="logs/${prefix}/${test_db_prefix}_pred_ep${ep}.log"
                  ${program} -mp ${mp} -trdb ${train_db_prefix} -tedb ${test_db_prefix} -trPro ${training_fn} -tePro ${testing_fn} -pv ${pro2vec_dic_fn} -do ${drop} -csv 1 -ms ${ms} -lr ${lr} -gpu ${gpu} -pr ${prefix} -pdfn ${testing_fn} -unit ${lstm_unit} -bs ${batch_size} -nf ${nf} -cv 1 -pd 0 -ep ${ep} -win ${win} -min ${min} >${log} 2>&1
                done
                done
                done
              done
              done
            done
          done
        done
      done
    done
  done
done

}


#testing_fn="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_Pid_Pseq_label.txt"
#test_db_prefix="SCRIBER_test"
#Predict ${testing_fn} ${test_db_prefix}
#
testing_fn="/home/j00492398/test_joey/interface-pred/Dataset/DS_72/DS_72_Pid_Pseq_label.txt"
test_db_prefix="DS_72"
Predict ${testing_fn} ${test_db_prefix}

testing_fn="/home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_Pseq_label.txt"
test_db_prefix="DS_164"
Predict ${testing_fn} ${test_db_prefix}

testing_fn="/home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_Pseq_label.txt"
test_db_prefix="DS_186"
Predict ${testing_fn} ${test_db_prefix}


#scriber reduced
testing_fn="/home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_reduced_355_Pid_Pseq_label.txt"
test_db_prefix="SCRIBER_test"
Predict ${testing_fn} ${test_db_prefix}