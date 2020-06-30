import os
import numpy as np
import sys
import math

def BuildFeatureDictionary(feature_np_1D):
    Feature_table = feature_np_1D
    max_Feature = np.amax(Feature_table)
    min_Feature = np.amin(Feature_table)
    # print("max_Feature: ", max_Feature)
    # print("min_Feature: ", min_Feature)
    normolized_Feature_table = (Feature_table - min_Feature) / (max_Feature - min_Feature)
    # print("normalized_Feature_table: ", normolized_Feature_table)

    Feature_dict = {}
    Feature_dict['A'] = normolized_Feature_table[0]
    Feature_dict['G'] = normolized_Feature_table[1]
    Feature_dict['V'] = normolized_Feature_table[2]
    Feature_dict['L'] = normolized_Feature_table[3]
    Feature_dict['I'] = normolized_Feature_table[4]
    Feature_dict['F'] = normolized_Feature_table[5]
    Feature_dict['Y'] = normolized_Feature_table[6]
    Feature_dict['W'] = normolized_Feature_table[7]
    Feature_dict['T'] = normolized_Feature_table[8]
    Feature_dict['S'] = normolized_Feature_table[9]
    Feature_dict['R'] = normolized_Feature_table[10]
    Feature_dict['K'] = normolized_Feature_table[11]
    Feature_dict['H'] = normolized_Feature_table[12]
    Feature_dict['D'] = normolized_Feature_table[13]
    Feature_dict['E'] = normolized_Feature_table[14]
    Feature_dict['N'] = normolized_Feature_table[15]
    Feature_dict['Q'] = normolized_Feature_table[16]
    Feature_dict['M'] = normolized_Feature_table[17]
    Feature_dict['P'] = normolized_Feature_table[18]
    Feature_dict['C'] = normolized_Feature_table[19]
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
    # array = np.array(
    # [[1.28,0.05,1.0,0.31,6.11,0.42,0.23],
    # [0.00,0.00,0.0,0.00,6.07,0.13,0.15],
    # [3.67,0.14,3.0,1.22,6.02,0.27,0.49],
    # [2.59,0.19,4.0,1.70,6.04,0.39,0.31],
    # [4.19,0.19,4.0,1.80,6.04,0.30,0.45],
    # [2.94,0.29,5.89,1.79,5.67,0.3,0.38],
    # [2.94,0.3,6.47,0.96,5.66,0.25,0.41],
    # [3.21,0.41,8.08,2.25,5.94,0.32,0.42],
    # [3.03,0.11,2.60,0.26,5.6,0.21,0.36],
    # [1.31,0.06,1.6,-0.04,5.7,0.20,0.28],
    # [2.34,0.29,6.13,-1.01,10.74,0.36,0.25],
    # [1.89,0.22,4.77,-0.99,9.99,0.32,0.27],
    # [2.99,0.23,4.66,0.13,7.69,0.27,0.3],
    # [1.6,0.11,2.78,-0.77,2.95,0.25,0.20],
    # [1.56,0.15,3.78,-0.64,3.09,0.42,0.21],
    # [1.6,0.13,2.95,-0.6,6.52,0.21,0.22],
    # [1.56,0.18,3.95,-0.22,5.65,0.36,0.25],
    # [2.35,0.22,4.43,1.23,5.71,0.38,0.32],
    # [2.67,0.0,2.72,0.72,6.8,0.13,0.34],
    # [1.77,0.13,2.43,1.54,6.35,0.17,0.41],
    # [0,0,0,0,0,0,0]]
    # )
    # print(array.shape)
    # for i in range(7):
    #     print(repr(array[:,i].reshape(21,)))
    # exit()
    feature1_np_1D = np.array(
        [1.28, 0., 3.67, 2.59, 4.19, 2.94, 2.94, 3.21, 3.03, 1.31, 2.34, 1.89, 2.99, 1.6, 1.56, 1.6, 1.56, 2.35, 2.67,1.77, 0.])
    feature2_np_1D = np.array([0.05, 0., 0.14, 0.19, 0.19, 0.29, 0.3, 0.41, 0.11, 0.06, 0.29,
                               0.22, 0.23, 0.11, 0.15, 0.13, 0.18, 0.22, 0., 0.13, 0.])
    feature3_np_1D = np.array([1., 0., 3., 4., 4., 5.89, 6.47, 8.08, 2.6, 1.6, 6.13,
                               4.77, 4.66, 2.78, 3.78, 2.95, 3.95, 4.43, 2.72, 2.43, 0.])
    feature4_np_1D = np.array([0.31, 0., 1.22, 1.7, 1.8, 1.79, 0.96, 2.25, 0.26,
                               -0.04, -1.01, -0.99, 0.13, -0.77, -0.64, -0.6, -0.22, 1.23,
                               0.72, 1.54, 0.])
    feature5_np_1D = np.array([6.11, 6.07, 6.02, 6.04, 6.04, 5.67, 5.66, 5.94, 5.6,
                               5.7, 10.74, 9.99, 7.69, 2.95, 3.09, 6.52, 5.65, 5.71,
                               6.8, 6.35, 0.])
    feature6_np_1D = np.array([0.42, 0.13, 0.27, 0.39, 0.3, 0.3, 0.25, 0.32, 0.21, 0.2, 0.36,
                               0.32, 0.27, 0.25, 0.42, 0.21, 0.36, 0.38, 0.13, 0.17, 0.])
    feature7_np_1D = np.array([0.23, 0.15, 0.49, 0.31, 0.45, 0.38, 0.41, 0.42, 0.36, 0.28, 0.25,
                               0.27, 0.3, 0.2, 0.21, 0.22, 0.25, 0.32, 0.34, 0.41, 0.])

    features = [feature1_np_1D, feature2_np_1D, feature3_np_1D, feature4_np_1D, feature5_np_1D, feature6_np_1D,
                feature7_np_1D]
    seq_fn = sys.argv[1]

    for i in range(len(features)):
        Feature_dict = BuildFeatureDictionary(features[i])
        out_fn = sys.argv[2] + str(i+1) + ".txt"
        load_fasta_and_compute(seq_fn, out_fn, Feature_dict)


if __name__ == '__main__':
    main()

