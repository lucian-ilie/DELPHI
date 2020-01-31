#!/bin/bash

database="/home/j00492398/test_joey/raw_features/ECO/uniprotDB/uniprot20_2015_06/uniprot20_2015_06"
hhblits="/home/j00492398/test_joey/raw_features/ECO/hh-suite/build/bin/hhblits"
inputpath=$1
outpath=$2

if [ ! -f ${outpath} ]; then
	mkdir ${outpath}
fi


for file in ${inputpath}/*.fasta;
do
	echo  ${outpath}/"$(basename "$file")"
	echo 'Checking'
	if [ ! -f ${outpath}/"$(basename "$file")" ]; then
		#echo $file $" has not been processed yet "
		${hhblits} -i $file -ohhm ${outpath}/"$(basename "$file")"  -d ${database}  -hide_cons -hide_pred -hide_dssp -v 0  -neffmax 1 -n 1
	#else
		#echo $file $" has already been processed "
	fi
done
