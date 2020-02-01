#!/bin/bash
#set -x
PRO_DIR="/home/j00492398/test_joey/interface-pred/"
program="python ./Many2One_BiLSTM.py"
pro2vec_dic_fn="${PRO_DIR}Features/protVec_100d_3grams.csv"
#training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_Pid_and_Pseq_label.txt"
training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_Pid_Pseq_label_0.4.txt"

#training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_Pid_and_Pseq_label_first20.txt"
#training_fn="${PRO_DIR}Dataset/survey/survey_train_nonRed_7426_Pid_and_Pseq_label.txt"
train_db_prefix="survey_train"
testing_fn="${PRO_DIR}Dataset/SCRIBER/SCRIBER_test_Pid_Pseq_label.txt"
test_db_prefix="SCRIBER_test"
#testing_fn="${PRO_DIR}Dataset/DS_164/DS_164_Pid_Pseq_label.txt"
#test_db_prefix="DS_164"
gpu=1

for pa in {4,}; do
for nf in {-1,}; do
  for lr in {0.001,}; do
    for ms in {85,}; do
      for batch_size in {1024,}; do
        for lstm_unit in {32,}; do
          for filter_size in {55,}; do
            for kernel_size in {6,}; do
          for win in {31,}; do
            for min in {50,}; do
              for ep in {250,}; do
                for drop in {0.5,0.7}; do
                  prefix="full_pa_${pa}_drop_${drop}_ms${ms}_lr${lr}_win${win}_min${min}_bs${batch_size}_nf${nf}_lstm${lstm_unit}_fil${filter_size}_ks${kernel_size}"
                  #prefix="debug"
                  mkdir -p logs/${prefix}
                  log="logs/${prefix}/train_ep${ep}.log"
                  ${program} -trdb ${train_db_prefix} -tedb ${test_db_prefix} -trPro ${training_fn} -tePro ${testing_fn} -pv ${pro2vec_dic_fn} -ks ${kernel_size} -fil ${filter_size} -do ${drop} -csv 1 -ms ${ms} -lr ${lr} -gpu ${gpu} -pr ${prefix} -pdfn ${testing_fn} -unit ${lstm_unit} -bs ${batch_size} -nf ${nf} -cv 1 -pd 0 -sm 0 -ld 1 -ep ${ep} -win ${win} -min ${min} -pa ${pa}>${log} 2>&1
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

for pa in {4,7}; do
for nf in {-1,}; do
  for lr in {0.001,}; do
    for ms in {7,}; do
      for batch_size in {1024,}; do
        for lstm_unit in {32,48}; do
          for filter_size in {32,}; do
            for kernel_size in {5,}; do
          for win in {31,}; do
            for min in {50,}; do
              for ep in {250,}; do
                for drop in {0.3,0.5,0.7}; do
                  prefix="full_pa_${pa}_drop_${drop}_ms${ms}_lr${lr}_win${win}_min${min}_bs${batch_size}_nf${nf}_lstm${lstm_unit}_fil${filter_size}_ks${kernel_size}"
                  #prefix="debug"
                  mkdir -p logs/${prefix}
                  log="logs/${prefix}/train_ep${ep}.log"
                  ${program} -trdb ${train_db_prefix} -tedb ${test_db_prefix} -trPro ${training_fn} -tePro ${testing_fn} -pv ${pro2vec_dic_fn} -ks ${kernel_size} -fil ${filter_size} -do ${drop} -csv 1 -ms ${ms} -lr ${lr} -gpu ${gpu} -pr ${prefix} -pdfn ${testing_fn} -unit ${lstm_unit} -bs ${batch_size} -nf ${nf} -cv 1 -pd 0 -sm 0 -ld 1 -ep ${ep} -win ${win} -min ${min} -pa ${pa}>${log} 2>&1
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

