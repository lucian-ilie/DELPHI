#!/bin/bash
# argv[1]: input file
# argv[2]: output dir
while read line
do
    if [[ ${line:0:1} == '>' ]]
    then
        outfile=${line#>}.fasta
        echo $line > $2$outfile
    else
        echo $line >> $2$outfile
    fi
done < $1