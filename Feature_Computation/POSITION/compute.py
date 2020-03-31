
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

def RetriveFeatureFromASequence(seq):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    Feature = []
    for index, item in enumerate(seq):
        Feature.append(float(index+1)/float(len(seq)))
    return Feature


def load_fasta_and_compute(seq_fn, out_fn):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        Feature = RetriveFeatureFromASequence(line_Pseq)
        fout.write(",".join(map(str,Feature)) + "\n")
    fin.close()
    fout.close()

def main():

    seq_fn = sys.argv[1]
    out_fn = sys.argv[2]
    load_fasta_and_compute(seq_fn, out_fn)



if __name__ == '__main__':
    main()

