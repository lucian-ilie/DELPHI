#!/bin/bash
inputpath=$1
outpath=$2


for file in ${inputpath}/*.fasta; 
do
	echo 'running cleanPSSM.sh'
	${PRO_DIR}/feature_computation/ECO/extractBlosum.sh $file > ${outpath}/"$(basename "$file")"
done	
