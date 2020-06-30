import os
import numpy as np
import time
import sys

def BuildFeatureDictionary():
    Feature_table = np.array(
        [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9,
         -1.3, 0.0])
    max_Feature = np.amax(Feature_table)
    min_Feature = np.amin(Feature_table)
    # print("max_Feature: ", max_Feature)
    # print("min_Feature: ", min_Feature)
    normolized_Feature_table = (Feature_table - min_Feature) / (max_Feature - min_Feature)
    # print("normalized_Feature_table: ", normolized_Feature_table)
    # normalized_Feature_table:
    # 0.7        0.77777778 0.11111111 0.11111111 0.81111111 0.45555556
    #  0.14444444 1.         0.06666667 0.92222222 0.71111111 0.11111111
    #  0.32222222 0.11111111 0.         0.41111111 0.42222222 0.96666667
    #  0.4        0.35555556 0.5

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

