#!/bin/bash
SHELL=/bin/bash
BASH_ENV=~/.bashrc_conda
source ~/.bashrc_conda
# This program minotors the DELPHI input directory every 5 mins with a cron job. 
INPUT_DIR=/work2/DELPHI_Server/input_dir
export PRO_DIR=/work2/DELPHI_Server/Src
scheulder_marker=/work2/DELPHI_Server/Src/delphi_is_running
cd ${PRO_DIR}
# $1: input_fn
function invoke_DELPHI(){
	cd ${PRO_DIR}
	if [ -f $1 ]; then
		# add a marker file so that only one DELPHI job can be run at a time
		touch ${scheulder_marker}
		CUR_DATE=$(date +%Y-%m-%d-%H-%M-%S)
		export TMP_DIR=${PRO_DIR}/tmp/tmp-${CUR_DATE}
		export OUT_DIR=${TMP_DIR}/out
		mkdir -p ${TMP_DIR}
		mkdir -p ${OUT_DIR}
		email=$( tail -n 1 $1)
		echo $email
		head -n -1 $1 > ${TMP_DIR}/input.fasta
		rm $1
		sleep 1
		cp ${TMP_DIR}/input.fasta ${OUT_DIR}/
		python3 /work2/DELPHI_Server/Src/utils/validate_input_sequence_server.py ${TMP_DIR}/input.fasta > ${OUT_DIR}/input_error_${CUR_DATE}.stderr
		if [ $? -ne 0 ]
		then
			cd ${OUT_DIR}
			tar czf result_${CUR_DATE}.tgz *.{stderr,fasta}
   			rsync  -r -l -K --progress ${OUT_DIR}/result_${CUR_DATE}.tgz yli922@gate.csd.uwo.ca:/home/yli922/public_html/directory_monitor/result_dir/
   			ssh yli922@gate.csd.uwo.ca "mail -a /home/yli922/public_html/directory_monitor/result_dir/result_${CUR_DATE}.tgz -s 'DELPHI results' 'yli922@uwo.ca' ${email} < /home/yli922/public_html/email_DELPHI_input_error.txt"
		else
			/work2/DELPHI_Server/Src/run_DELPHI.sh ${TMP_DIR}/input.fasta > ${OUT_DIR}/delphi.log 2>&1
			cd ${OUT_DIR}
			tar czf result_${CUR_DATE}.tgz *.{txt,log,fasta}
			rsync  -r -l -K --progress ${OUT_DIR}/result_${CUR_DATE}.tgz yli922@gate.csd.uwo.ca:/home/yli922/public_html/directory_monitor/result_dir/
			ssh yli922@gate.csd.uwo.ca "mail -a /home/yli922/public_html/directory_monitor/result_dir/result_${CUR_DATE}.tgz -s 'DELPHI results' 'yli922@uwo.ca' ${email} < /home/yli922/public_html/email_DELPHI.txt"
		fi
		rm ${scheulder_marker}
	else
		echo "$1 doesn't exist"
	fi
}
#if nothing in the input directory, exit directly
files=$(find ${INPUT_DIR} -name "*.input")
if [ -z "$files" ]
then
      echo "\$files is empty"
      exit
fi

# wait first for all running jobs to finish
while true; do
	if [ -f ${scheulder_marker} ]; then
		echo "[Info:] Running DELPHI job(s) detected, wait..."
		date
		sleep 10m
	else
		break
	fi
done
echo "[Info:] Start running a new DELPHI job now"
date
files=$(find ${INPUT_DIR} -name "*.input")
if [ -z "$files" ]
then
      echo "\$files is empty"
else
    for file in $files
	do
		echo $file
		invoke_DELPHI $file
	done 
fi
