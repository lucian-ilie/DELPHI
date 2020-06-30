import os
import numpy as np
import sys

def BuildFeatureDictionary():
    Feature_table = np.array([7.00, 7.00, 3.65, 3.22, 7.00, 7.00, 6.00, 7.00, 10.53, 7.00,
                              7.00, 8.18, 7.00, 7.00, 12.48, 7.00, 7.00, 7.00, 7.00, 10.07, 7.0])
    max_Feature = np.amax(Feature_table)
    min_Feature = np.amin(Feature_table)
    # print("max_Feature: ", max_Feature)
    # print("min_Feature: ", min_Feature)
    normolized_Feature_table = (Feature_table - min_Feature) / (max_Feature - min_Feature)
    # print("normalized_Feature_table: ", normolized_Feature_table)
    # normalized_Feature_table:
    # 0.40820734 0.40820734 0.04643629 0.         0.40820734 0.40820734
    #  0.30021598 0.40820734 0.78941685 0.40820734 0.40820734 0.53563715
    #  0.40820734 0.40820734 1.         0.40820734 0.40820734 0.40820734
    #  0.40820734 0.73974082 0.40820734

    Feature_dict = {}
    Feature_dict['A'] = normolized_Feature_table[0]
    Feature_dict['C'] = normolized_Feature_table[1]
    Feature_dict['D'] = normolized_Feature_table[2]
    Feature_dict['E'] = normolized_Feature_table[3]
    Feature_dict['F'] = normolized_Feature_table[4]
    Feature_dict['G'] = normolized_Feature_table[5]
    Feature_dict['H'] = normolized_Feature_table[6]
    Feature_dict['I'] = normolized_Feature_table[7]
    Feature_dict['K'] = normolized_Feature_table[8]
    Feature_dict['L'] = normolized_Feature_table[9]
    Feature_dict['M'] = normolized_Feature_table[10]
    Feature_dict['N'] = normolized_Feature_table[11]
    Feature_dict['P'] = normolized_Feature_table[12]
    Feature_dict['Q'] = normolized_Feature_table[13]
    Feature_dict['R'] = normolized_Feature_table[14]
    Feature_dict['S'] = normolized_Feature_table[15]
    Feature_dict['T'] = normolized_Feature_table[16]
    Feature_dict['V'] = normolized_Feature_table[17]
    Feature_dict['W'] = normolized_Feature_table[18]
    Feature_dict['Y'] = normolized_Feature_table[19]
    Feature_dict['X'] = normolized_Feature_table[20]

    return Feature_dict



def GetFeature(AA, Feature_dict):
    if (AA not in Feature_dict):
        print("[warning]: Feature_dict can't find ", AA, ". Returning 0")
        return 0
    else:
        return Feature_dict[AA]

def RetriveFeatureFromASequence(seq, Feature_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    Feature = []
    for index, item in enumerate(seq):
        Feature.append(GetFeature(item, Feature_dict))
    return Feature


def load_fasta_and_compute(seq_fn, out_fn, Feature_dict):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        Feature = RetriveFeatureFromASequence(line_Pseq, Feature_dict)
        fout.write(",".join(map(str,Feature)) + "\n")
    fin.close()
    fout.close()

def main():
    Feature_dict = BuildFeatureDictionary()
    seq_fn = sys.argv[1]
    out_fn = sys.argv[2]
    load_fasta_and_compute(seq_fn, out_fn, Feature_dict)



if __name__ == '__main__':
    main()

