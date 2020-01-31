
################################################
# fix the random see value so the results are re-producible
seed_value = 7
import os
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)
###############################################

import csv
import logging
import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Reshape, TimeDistributed, Bidirectional, CuDNNLSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow.python.keras.callbacks import TensorBoard
import time
import sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight
from itertools import chain
import argparse
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

def BuildFeatureDictionary():
    Feature_table = np.array([7.00, 7.00, 3.65, 3.22, 7.00, 7.00, 6.00, 7.00, 10.53, 7.00,
                              7.00, 8.18, 7.00, 7.00, 12.48, 7.00, 7.00, 7.00, 7.00, 10.07, 7.0])
    max_Feature = np.amax(Feature_table)
    min_Feature = np.amin(Feature_table)
    print("max_Feature: ", max_Feature)
    print("min_Feature: ", min_Feature)
    normolized_Feature_table = (Feature_table - min_Feature) / (max_Feature - min_Feature)
    print("normalized_Feature_table: ", normolized_Feature_table)
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


def load_fasta_and_compute(seq_fn, ds_prefix):
    fin = open(seq_fn, "r")
    # fout = open(out_fn, "w")
    while True:
        Pid = fin.readline().rstrip("\n")[1:]
        line_Pseq = fin.readline().rstrip("\n")
        if not line_Pseq:
            break
        pssm_fn = "/home/j00492398/test_joey/raw_features/PSSM/" + ds_prefix + "/" + Pid + ".pssm"
        LoadPSSMandPrintFeature(pssm_fn, ds_prefix, Pid, line_Pseq)
        # fout.write(line_Pid)
        # fout.write(line_Pseq)

        # fout.write(",".join(map(str,Feature)) + "\n")
    fin.close()
    # fout.close()


def extract_lines(pssmFile):
    fin = open(pssmFile)
    pssmLines = []
    if fin == None:
        return
    for i in range(3):
        fin.readline()  # exclude the first three lines
    while True:
        psspLine = fin.readline()
        if psspLine.strip() == '' or psspLine.strip() == None:
            break
        pssmLines.append(psspLine)
    fin.close()
    return pssmLines



def LoadPSSMandPrintFeature(pssm_fn, ds_prefix, Pid, line_Pseq):
    # print(Pid)
    global min_value, max_value
    fin = open(pssm_fn, "r")
    pssmLines=extract_lines(pssm_fn)
    # print(pssmLines)
    seq_len = len(pssmLines)
    if (seq_len == len(line_Pseq)):
        # exit(1)
        pssm_np_2D = np.zeros(shape=(20, seq_len))
        for i in range(seq_len):

            # fist 69 chars are what we need
            # print(pssmLines[i][9:70])
            #
            values_20 = pssmLines[i].split()[2:22]
            # print(values_20)
            for aa_index in range(20):
                pssm_np_2D[aa_index][i] = (float(values_20[aa_index]) - min_value)/(max_value - min_value)
                # max_value = max(max_value, float(values_20[aa_index]))
                # min_value = min(min_value, float(values_20[aa_index]))
        fin.close()

        # print to feature file
        for i in range(1, 21):
            out_fn = "/home/j00492398/test_joey/interface-pred/Features/PSSM/" + ds_prefix + "_" + str(i) + ".txt"
            fout = open(out_fn, "a+")
            fout.write(">" + Pid + "\n")
            fout.write(line_Pseq + "\n")
            fout.write(",".join(map(str,pssm_np_2D[i-1])) + "\n")
            fout.close()
    else:
        print("length doesn't match for protein ", Pid)
        print("PSSM file has ", seq_len," lines, but sequence length is ", len(line_Pseq))
def main():
    # Feature_dict = BuildFeatureDictionary()
    seq_fn = sys.argv[1]
    out_base_fn = sys.argv[2]
    # LoadPSSMandPrintFeature("/home/j00492398/test_joey/interface-pred/workspace/compute_feature_dictionary/PSSM/2WD5A.pssm","test")
    # exit(1)
    load_fasta_and_compute(seq_fn, out_base_fn)
    print("max_value: ", max_value)
    print("min_value: ", min_value)

max_value = 13.0
min_value = -16.0
if __name__ == '__main__':
    main()

