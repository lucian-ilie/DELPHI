#!/bin/bash
#set -x

run_ds() {
  DS=$1
  INPUT_FN=$2
  mkdir -p ${DS}_temp
cp /home/j00492398/test_joey/interface-pred/Features/Anchor/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/Anchor.txt
cp /home/j00492398/test_joey/interface-pred/Features/ECO/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/ECO.txt
cp /home/j00492398/test_joey/interface-pred/Features/HSP/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/HSP.txt
cp /home/j00492398/test_joey/interface-pred/Features/HYD/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/HYD.txt
for i in {1,2,3}; do
  cp /home/j00492398/test_joey/interface-pred/Features/PHY_Char/${DS}_${i}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/PHY_Char${i}.txt
done
for i in {1,2,3,4,5,6,7}; do
cp /home/j00492398/test_joey/interface-pred/Features/PHY_Prop/${DS}_${i}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/PHY_Prop${i}.txt
  done
cp /home/j00492398/test_joey/interface-pred/Features/PKA/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/PKA.txt
cp /home/j00492398/test_joey/interface-pred/Features/POSITION/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/POSITION.txt
cp /home/j00492398/test_joey/interface-pred/Features/Pro2Vec_1D/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/Pro2Vec_1D.txt
for i in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; do
cp /home/j00492398/test_joey/interface-pred/Features/PSSM/${DS}_${i}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/PSSM${i}.txt
done
cp /home/j00492398/test_joey/interface-pred/Features/RAA/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/RAA.txt
cp /home/j00492398/test_joey/interface-pred/Features/RSA/${DS}.txt /home/j00492398/test_joey/DELPHI_server_src/${DS}_temp/RSA.txt

python3 predict.py -i $INPUT_FN -d ${DS}_temp -o out_${DS} -c 1

}
# run_ds DS_164 /home/j00492398/test_joey/interface-pred/Dataset/DS_164/DS_164_Pid_and_Pseq.txt
#run_ds DS_186 /home/j00492398/test_joey/interface-pred/Dataset/DS_186/DS_186_Pid_and_Pseq.txt
#run_ds SCRIBER_test /home/j00492398/test_joey/interface-pred/Dataset/SCRIBER/SCRIBER_test_Pid_and_Pseq.txt

# build db
cd /work2/DELPHI_Server/PSSM_database/PSSMs
python3 /work2/DELPHI_Server/Src/utils/build_PSSM_DB.py /work2/DELPHI_Server/PSSM_database/PSSMs/ 

# performance evaluation
python3 /work2/DELPHI_Server/Src/utils/performance_evaluation.py out_DS72 dataset/DS_72_Pid_Pseq_label.txt ds72_delphi_cpu_server












