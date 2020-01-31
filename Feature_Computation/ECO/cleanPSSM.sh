inputpath=$1
outpath=$2


for file in ${inputpath}/*.fasta; 
do
	echo 'processing'
	/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/ECO/extractBlosum.sh $file > ${outpath}/"$(basename "$file")"
done	
