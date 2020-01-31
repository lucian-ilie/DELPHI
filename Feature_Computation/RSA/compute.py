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


# dbDir is where RSA database. The DIR of where the RSA files should be loaded, remember to add '/'
def LoadRSA(seq_fn, dbDir, out_fn, max_rsa, min_rsa):
    fin_seq = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_PID = fin_seq.readline().rstrip("\n")[1:]
        line_Seq = fin_seq.readline().rstrip("\n")
        if not line_Seq:
            break

        rsa = []
        rsa_fn = dbDir + "/asaq." + line_PID + ".fasta/rasaq.pred"
        try:
            fin = open(rsa_fn, "r")
        except Exception as e:
            print("open file failed. exit now: ", rsa_fn)
            exit(1)

        lines = fin.readlines()
        seq_len = len(line_Seq)
        for x in lines:
            value = float(x.split(' ')[2])
            value = (value - min_rsa) / (max_rsa - min_rsa)
            rsa.append(value)

        fin.close()
        # some proteins' rsa file fails to be generated, pad with 0
        if (len(rsa) == 0):
            rsa = [0] * seq_len
            print("[warning:]", line_PID, "has no RSA file. Pad ", seq_len, " 0 for it.")
        fout.write(">" + line_PID + "\n")
        fout.write(line_Seq + "\n")
        fout.write(",".join(map(str, rsa)) + "\n")
    fin_seq.close()
    fout.close()


def main():
    seq_fn = sys.argv[1]
    raw_rsa_dir = sys.argv[2]
    out_fn = sys.argv[3]

    LoadRSA(seq_fn, raw_rsa_dir, out_fn, 0.5534, -0.9951)


if __name__ == '__main__':
    main()
