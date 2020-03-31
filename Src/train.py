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

# in training feature file
# 1: Protein ID and type of binding
# 2: Amino acid sequence
# 3: Space-separated RAA values for each residue
# 4: Space-separated native RSA values for each residue (* means that the value could not be computed since structure of this residue was missing in the PDB file)
# 5: Space-separated putative RSA values for each residue
# 6: Space-separated ECO value for each residue, comma separated (use this first)

# features will be calculated by myself:


# print numpy array without elimination
# np.set_printoptions(threshold=sys.maxsize)


def CalculateEvaluationMetrics(y_true, y_pred):
    TP = float(0)
    FP = float(0)
    TN = float(0)
    FN = float(0)
    for i, j in zip(y_true, y_pred):
        if (i == 1 and j == 1):
            TP += 1
        elif (i == 0 and j == 1):
            FP += 1
        elif (i == 0 and j == 0):
            TN += 1
        elif (i == 1 and j == 0):
            FN += 1
    print("TP: ", TP)
    print("FP: ", FP)
    print("TN: ", TN)
    print("FN: ", FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    # same as sensitivity
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("MCC: ", MCC)
    return TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC


# dim: delimiter
def get_array_of_float_from_a_line(line, dim):
    res = []
    line = line.rstrip('\n').rstrip(' ').split(dim)
    # print (line)
    res += [float(i) for i in line]
    return res


# letter/'1' means interface; '.'/'0' means non-interface
def get_array_of_int_from_a_line(line):
    res = []
    for i in line.rstrip('\n').rstrip(' '):
        if (i == '.' or i == '0'):
            res.append(0)
        else:
            res.append(1)
    return res


# split a 1D list into window size, the last piece that's shorter than window size is discarded
# TODO: check if padding with non-0 is better for accuracy
def Split1Dlist2NpArrays(args, inputList):
    win_size = args.window_size
    assert (win_size <= args.mim_seq_len)
    sta = 0
    index = 0
    list_1d_after_split = []
    while (sta + win_size <= len(inputList)):
        index += 1
        list_1d_after_split.extend(inputList[sta:sta + win_size])
        sta = (sta + int(win_size / 2))
    # print(list_1d_after_split)
    if (int(len(inputList) / (win_size / 2)) - 1 != index):
        print("win_size: ", win_size)
        print("pro_len: ", len(inputList))
        print("2*int(len(inputList)/win_size - 1: ", int(len(inputList) / (win_size / 2)) - 1)
        print("index: ", index)
    # assert(2*int(len(inputList)/win_size) == index)
    return list_1d_after_split, index

    # win_size = args.window_size
    # input_1D_array = np.array(inputList)
    # assert (win_size <= args.mim_seq_len)
    # sta = 0
    # np_1d_array = np.array([])
    # index = 0
    # while (sta + win_size <= input_1D_array.size):
    #     index += 1
    #     np_1d_array = np.append(np_1d_array, input_1D_array[sta:sta + win_size])
    #     sta = (sta + int(win_size / 2))
    #     # print("sta: ", sta)
    # # print("index: ", index)
    # # np_2d_array = np_1d_array.reshape(index, win_size)
    # # print("shape: ", np_2d_array.shape)
    # # print("np_2d_array: \n", np_2d_array)
    # return np_1d_array, index


# split input 2D list
def Split2DList2NpArrays(args, input2DList):
    total_num_of_rows = 0
    input2DList_after_split = []
    for input1DList in input2DList:
        if (len(input1DList) >= args.mim_seq_len):
            input1DList_after_split, num_of_rows_in_input1DList = Split1Dlist2NpArrays(args, input1DList)
            input2DList_after_split.extend(input1DList_after_split)
            total_num_of_rows += num_of_rows_in_input1DList
        else:
            logging.info("ignoring sequence with length: %d", len(input1DList))
    np_2d_array = np.asarray(input2DList_after_split)
    np_2d_array = np_2d_array.reshape(total_num_of_rows, args.window_size)
    # print("np_2d_array.shape: ", np_2d_array.shape)
    # print("np_2d_array: \n", np_2d_array)
    return np_2d_array


# csvPre is the thing to put in front of csv prefix
def PlotRocAndPRCurvesAndMetrics(truth, pred, args, csvPre=""):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    # print("fpr: ", fpr)
    # print("tpr: ", tpr)
    # print("thresholds: ", thresholds)

    # ROC curve
    au_roc = auc(fpr, tpr)
    print("Area under ROC curve: ", au_roc)
    plt.title(args.prefix + ' ROC curve')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # plt.show()
    plt.savefig("plots/" + args.prefix + "/ROC_" + csvPre + ".pdf")
    plt.close()

    # PR curve
    precision, recall, thresholds = precision_recall_curve(truth, pred)
    aupr = auc(recall, precision)
    print("Area under PR curve: ", aupr)
    plt.title(args.prefix + ' PR curve')
    plt.plot(recall, precision)
    # plt.show()
    plt.savefig("plots/" + args.prefix + "/PR_" + csvPre + ".pdf")
    plt.close()

    # evaluation metrics
    # step 1: calculate the threshold then convert score to binary number
    sorted_pred = np.sort(pred)
    sorted_pred_descending = np.flip(sorted_pred)  # from big to small
    num_of_1 = np.count_nonzero(truth)
    threshold = sorted_pred_descending.item(num_of_1 - 1)
    pred_binary = np.where(pred >= threshold, 1, 0)
    TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC = CalculateEvaluationMetrics(truth, pred_binary)
    if (int(args.print_csv) == 1):
        PrintToCSV(csvPre + args.prefix, au_roc, aupr, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC,
                   threshold)
    return au_roc


# split a 2D Np array (100 * pro_len) using a 2D sliding window. return a 3D np array
def Split2DNp3DNp(args, input2DNp):
    # print("input2DNp.shape ", input2DNp.shape)
    # print("input2dNp: ")
    # print(input2DNp)
    num_rows = int(input2DNp.shape[1] / (args.window_size / 2)) - 1
    win_size = args.window_size
    res = np.zeros((num_rows, win_size, 100))
    assert (win_size <= args.mim_seq_len)
    sta = 0
    row_num = 0
    # list_1d_after_split = []
    while (sta + win_size <= input2DNp.shape[1]):
        slice_of_2D_np = input2DNp[:, sta:sta + win_size]
        slice_of_2D_np = np.swapaxes(slice_of_2D_np, 0, 1).reshape(1, win_size, 100)
        # if (row_num == 0):
        #     print("slice_of_2D_np.shape: ", slice_of_2D_np.shape)
        #     # print("first row input2DNp[:, sta:sta + win_size]",input2DNp[:, sta:sta + win_size])
        #     print("first row: slice_of_2D_np")
        #     print(slice_of_2D_np[:, :, 0])
        # print("slice_of_2D_np.shape ", slice_of_2D_np.shape)
        res[row_num, :, :] = slice_of_2D_np
        sta = (sta + int(win_size / 2))
        row_num += 1
    # print("res in Split2DNp3DNp first feature")
    # print("res.shape", res.shape)
    # print(res[:, :, 0])
    return res
    #     list_1d_after_split.extend(inputList[sta:sta + win_size])
    #     sta = (sta + int(win_size / 2))
    # # print(list_1d_after_split)
    # return list_1d_after_split, index


# pro2vec: list of 2D np arrays
# return np 3D arrays after splitting
def SplitPro2Vec2NpArrays(args, pro2vec, num_row, num_col):
    res = np.zeros((num_row, num_col, 100))
    row_index = 0
    progress = 0
    for input2DNp in pro2vec:
        if (progress % 100 == 0):
            logging.info("SplitPro2Vec2NpArrays processing %dth protein..", progress)
            progress += 1
        if (input2DNp[0].size) >= args.mim_seq_len:
            pro2vec_3DNp = Split2DNp3DNp(args, input2DNp)

            res[row_index:row_index + pro2vec_3DNp.shape[0], :, :] = pro2vec_3DNp
            row_index = row_index + pro2vec_3DNp.shape[0]
            # res = np.append(res, pro2vec_3DNp, axis=2) if res.size else pro2vec_3DNp
    return res
    # total_num_of_rows = 0
    # input2DList_after_split = []
    # for input1DList in input2DList:
    #     if (len(input1DList) >= args.mim_seq_len):
    #         input1DList_after_split, num_of_rows_in_input1DList = Split1Dlist2NpArrays(args, input1DList)
    #         input2DList_after_split.extend(input1DList_after_split)
    #         total_num_of_rows += num_of_rows_in_input1DList
    #     else:
    #         logging.info("ignoring sequence with length: %d", len(input1DList))
    # np_2d_array = np.asarray(input2DList_after_split)
    # np_2d_array = np_2d_array.reshape(total_num_of_rows, args.window_size)
    # print("np_2d_array.shape: ", np_2d_array.shape)
    # # print("np_2d_array: \n", np_2d_array)
    # return np_2d_array

    # feature 1, 2, 3 are list of lists
    # pro2vec is list of 2D np arrays


def Convert2DListTo3DNp(args, list2D):
    np_2d = Split2DList2NpArrays(args, list2D)
    np_3D = np_2d.reshape(np_2d.shape[0], np_2d.shape[1], 1)
    return np_3D


def PredictUsingModel1(args, model_protein, model_DNA, model_RNA, model_ligand, all_features):
    protein_pred = model_protein.predict(all_features, batch_size=args.batch_size)
    DNA_pred = model_DNA.predict(all_features, batch_size=args.batch_size)
    RNA_pred = model_RNA.predict(all_features, batch_size=args.batch_size)
    ligand_pred = model_ligand.predict(all_features, batch_size=args.batch_size)

    # 2D to 3D
    protein_pred = protein_pred.reshape(protein_pred.shape[0], protein_pred.shape[1], 1)
    DNA_pred = DNA_pred.reshape(DNA_pred.shape[0], DNA_pred.shape[1], 1)
    RNA_pred = RNA_pred.reshape(RNA_pred.shape[0], RNA_pred.shape[1], 1)
    ligand_pred = ligand_pred.reshape(ligand_pred.shape[0], ligand_pred.shape[1], 1)

    return protein_pred, DNA_pred, RNA_pred, ligand_pred


def ComputeOtherTwoInputsForModel2(protein_pred, DNA_pred, RNA_pred, ligand_pred):
    temp = np.maximum(DNA_pred, RNA_pred)
    max_among_three = np.maximum(temp, ligand_pred)
    pro_minus_max = protein_pred - max_among_three
    return max_among_three, pro_minus_max


def TrainModel(ECO, RAA, putativeRSA, ECO_calculated, RAA_calculated, normalized_RSA, pro2vec, label_protein, args,
               ECO_testing, RAA_testing, RSA_testing, pro2vec_testing, label_testing, label_DNA, label_RNA,
               label_ligand):
    logging.info("converting features and label from list to np array")
    ECO_3D_np = Convert2DListTo3DNp(args, ECO)
    RAA_3D_np = Convert2DListTo3DNp(args, RAA)
    RSA_3D_np = Convert2DListTo3DNp(args, putativeRSA)
    normalized_RSA_3D_np = Convert2DListTo3DNp(args, normalized_RSA)
    ECO_calculated_3D_np = Convert2DListTo3DNp(args, ECO_calculated)
    RAA_calculated_3D_np = Convert2DListTo3DNp(args, RAA_calculated)

    ECO_testing_3D_np = Convert2DListTo3DNp(args, ECO_testing)
    RAA_testing_3D_np = Convert2DListTo3DNp(args, RAA_testing)
    RSA_testing_3D_np = Convert2DListTo3DNp(args, RSA_testing)

    label_protein_3D_np = Convert2DListTo3DNp(args, label_protein)
    label_DNA_3D_np = Convert2DListTo3DNp(args, label_DNA)
    label_RNA_3D_np = Convert2DListTo3DNp(args, label_RNA)
    label_ligand_3D_np = Convert2DListTo3DNp(args, label_ligand)

    label_testing_3D_np = Convert2DListTo3DNp(args, label_testing)
    assert (normalized_RSA_3D_np.shape == ECO_calculated_3D_np.shape == RAA_calculated_3D_np.shape
            == label_protein_3D_np.shape == label_DNA_3D_np.shape == label_RNA_3D_np.shape == label_ligand_3D_np.shape)
    assert (ECO_testing_3D_np.shape == RAA_testing_3D_np.shape == RSA_testing_3D_np.shape == label_testing_3D_np.shape)
    # get the shape of a 1D feature for pro2Vec to use
    num_row = ECO_3D_np.shape[0]
    num_col = ECO_3D_np.shape[1]
    num_row_testing = ECO_testing_3D_np.shape[0]
    num_col_testing = ECO_testing_3D_np.shape[1]

    pro2vec_3D_np = SplitPro2Vec2NpArrays(args, pro2vec, num_row, num_col)
    pro2vec_testing_3D_np = SplitPro2Vec2NpArrays(args, pro2vec_testing, num_row_testing, num_col_testing)

    # if (args.num_feature == 0):
    #     logging.info("Using 3 features provided by the paper")
    #     all_features_3D_np = np.concatenate((ECO_3D_np, RAA_3D_np, RSA_3D_np), axis=2)
    if (args.num_feature == -1):
        logging.info("Using 3 features calculated by me")
        all_features_3D_np = np.concatenate((normalized_RSA_3D_np, ECO_calculated_3D_np,
                                             RAA_calculated_3D_np), axis=2)
        all_testing_features_3D_np = np.concatenate((RSA_testing_3D_np, ECO_testing_3D_np,
                                                     RAA_testing_3D_np), axis=2)
    # elif (args.num_feature == 1):  # ECO
    #     logging.info("Using provided ECO only")
    #     all_features_3D_np = ECO_3D_np
    # elif (args.num_feature == 2):  # RAA
    #     logging.info("Using provided RAA only")
    #     all_features_3D_np = RAA_3D_np
    # elif (args.num_feature == 3):  # putativeRSA
    #     logging.info("Using provided RSA only")
    #     all_features_3D_np = RSA_3D_np
    elif (args.num_feature == 4):
        logging.info("Using computed RSA only")
        all_features_3D_np = normalized_RSA_3D_np
        all_testing_features_3D_np = RSA_testing_3D_np
    elif (args.num_feature == 5):
        logging.info("Using computed ECO only")
        all_features_3D_np = ECO_calculated_3D_np
        all_testing_features_3D_np = ECO_testing_3D_np
    elif (args.num_feature == 6):
        logging.info("Using computed RAA only")
        all_features_3D_np = RAA_calculated_3D_np
        all_testing_features_3D_np = RAA_testing_3D_np
    # elif (args.num_feature == 103):
    #     logging.info("Using 3 features provided by the paper + pro2vec")
    #     all_features_3D_np = np.concatenate((ECO_3D_np, RAA_3D_np, RSA_3D_np, pro2vec_3D_np), axis=2)
    elif (args.num_feature == 100):
        logging.info("Using pro2vec only")
        all_features_3D_np = pro2vec_3D_np
        all_testing_features_3D_np = pro2vec_testing_3D_np
    elif (args.num_feature == 203):
        logging.info("Using 3 features calculated by me + pro2vec")
        all_features_3D_np = np.concatenate((normalized_RSA_3D_np, ECO_calculated_3D_np,
                                             RAA_calculated_3D_np, pro2vec_3D_np), axis=2)
        all_testing_features_3D_np = np.concatenate((RSA_testing_3D_np, ECO_testing_3D_np,
                                                     RAA_testing_3D_np, pro2vec_testing_3D_np), axis=2)
    else:
        logging.error("option nb --num_feature is invalid")

    cur_time = time.time()
    cur_time_formatted = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    tensorboard = TensorBoard(log_dir="tensorboard_log/{}".format(args.prefix))

    print("all_features_3D_np.shape: ", all_features_3D_np.shape)

    # cross validation
    kfold = KFold(n_splits=5, shuffle=True)
    split_idx = 0
    print("all_features_3D_np.shape: ", all_features_3D_np.shape)
    print("label_3D_np.shape: ", label_protein_3D_np.shape)
    print("all_testing_features_3D_np.shape: ", all_testing_features_3D_np.shape)
    print("label_testing_3D_np.shape: ", label_testing_3D_np.shape)
    logging.info("Doing cross validation")
    AUC = []
    for train, test in kfold.split(all_features_3D_np):
        model_protein = BuildModel(args)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
        mc = ModelCheckpoint("savedModel/" + args.prefix + "/" + "best_model.h5", monitor='val_acc', mode='max',
                             verbose=1, save_best_only=True)
        if (split_idx == 0):
            model_protein.summary()
        # model_DNA = BuildModel(args)
        # model_RNA = BuildModel(args)
        # model_ligand = BuildModel(args)
        split_idx += 1
        if (split_idx > 1):
            continue
        print("split_idx: ", split_idx)
        print("train: ", train)
        print("test: ", test)
        print("train_feature.shape: ", all_features_3D_np[train].shape)
        print("test_feature.shape: ", all_features_3D_np[test].shape)
        print("train_label.shape: ", label_protein_3D_np[train].shape)
        print("test_label.shape: ", label_protein_3D_np[test].shape)
        logging.info("Start training model1...")

        # class_weights={0:1.0,
        #               1:5.0}
        class_weights_protein = class_weight.compute_class_weight('balanced',
                                                                  np.unique(label_protein_3D_np.ravel()),
                                                                  label_protein_3D_np.ravel())
        class_weights_DNA = class_weight.compute_class_weight('balanced',
                                                              np.unique(label_DNA_3D_np.ravel()),
                                                              label_DNA_3D_np.ravel())
        class_weights_RNA = class_weight.compute_class_weight('balanced',
                                                              np.unique(label_RNA_3D_np.ravel()),
                                                              label_RNA_3D_np.ravel())
        class_weights_ligand = class_weight.compute_class_weight('balanced',
                                                                 np.unique(label_ligand_3D_np.ravel()),
                                                                 label_ligand_3D_np.ravel())
        print("class_weights_protein:", class_weights_protein)
        print("class_weights_DNA:", class_weights_DNA)
        print("class_weights_RNA:", class_weights_RNA)
        print("class_weights_ligand:", class_weights_ligand)

        # print("label.unique(): ",np.unique(label_protein_3D_np[train].reshape(-1, 16).ravel()))
        log_time("Start training model1 protein...")
        CV_train_history_protein = model_protein.fit(all_features_3D_np[train],
                                                     label_protein_3D_np[train].reshape(-1, args.window_size),
                                                     callbacks=[tensorboard, es], shuffle=True,
                                                     batch_size=args.batch_size, class_weight=class_weights_protein,
                                                     epochs=args.epochs, verbose=2, validation_data=(
                all_features_3D_np[test], label_protein_3D_np[test].reshape(-1, args.window_size)))
        '''
        log_time("Start training model1 DNA...")
        CV_train_history_DNA = model_DNA.fit(all_features_3D_np[train],
                                             label_DNA_3D_np[train].reshape(-1, 16), batch_size=args.batch_size,
                                             class_weight=class_weights_DNA,
                                             epochs=args.epochs, verbose=2, validation_data=(
                all_features_3D_np[test], label_DNA_3D_np[test].reshape(-1, 16)))
        log_time("Start training model1 RNA...")
        CV_train_history_RNA = model_RNA.fit(all_features_3D_np[train],
                                             label_RNA_3D_np[train].reshape(-1, 16), batch_size=args.batch_size,
                                             class_weight=class_weights_RNA,
                                             epochs=args.epochs, verbose=2, validation_data=(
                all_features_3D_np[test], label_RNA_3D_np[test].reshape(-1, 16)))
        log_time("Start training model1 ligand...")
        CV_train_history_ligand = model_ligand.fit(all_features_3D_np[train],
                                                   label_ligand_3D_np[train].reshape(-1, 16),
                                                   batch_size=args.batch_size, class_weight=class_weights_ligand,
                                                   epochs=args.epochs, verbose=2, validation_data=(
                all_features_3D_np[test], label_ligand_3D_np[test].reshape(-1, 16)))

        # 2D_np to 3D_np
        # args, model_protein, model_DNA, model_RNA, model_ligand, all_features
        protein_pred, DNA_pred, RNA_pred, ligand_pred = PredictUsingModel1(args, model_protein, model_DNA, model_RNA,
                                                                           model_ligand, all_features_3D_np[train])
        '''
        # check the AUC before applying model2
        log_time("Start CV testing...")
        protein_pred_on_CV_test_before_model2 = model_protein.predict(all_features_3D_np[test],
                                                                      batch_size=args.batch_size)
        print("protein_pred_on_CV_test_before_model2.shape: ", protein_pred_on_CV_test_before_model2.shape)
        PlotRocAndPRCurvesAndMetrics(label_protein_3D_np[test].ravel(), protein_pred_on_CV_test_before_model2.ravel(),
                                     args, "CV_test_" + str(split_idx) + "_")

        log_time("Start testing...")
        y_pred_testing = model_protein.predict(all_testing_features_3D_np, batch_size=args.batch_size).ravel()
        if (int(args.plot_curves) == 1):
            PlotRocAndPRCurvesAndMetrics(label_testing_3D_np.ravel(), y_pred_testing.ravel(), args, "Testing_")

        '''
        pred_on_CV_test_before_model2 = model_DNA.predict(all_features_3D_np[test], batch_size=args.batch_size)
        PlotRocAndPRCurvesAndMetrics(label_DNA_3D_np[test].ravel(), pred_on_CV_test_before_model2.ravel(), args,
                                     "CV_train_DNA_before_model2_")

        pred_on_CV_test_before_model2 = model_RNA.predict(all_features_3D_np[test], batch_size=args.batch_size)
        PlotRocAndPRCurvesAndMetrics(label_RNA_3D_np[test].ravel(), pred_on_CV_test_before_model2.ravel(), args,
                                     "CV_train_RNA_before_model2_")

        pred_on_CV_test_before_model2 = model_ligand.predict(all_features_3D_np[test], batch_size=args.batch_size)
        PlotRocAndPRCurvesAndMetrics(label_ligand_3D_np[test].ravel(), pred_on_CV_test_before_model2.ravel(), args,
                                     "CV_train_ligand_before_model2_")
        
        
        model2 = BuildSecondModel(args)
        
        max_among_three, pro_minus_max = ComputeOtherTwoInputsForModel2(protein_pred, DNA_pred, RNA_pred, ligand_pred)
        all_features_model2_3D_np = np.concatenate((protein_pred, DNA_pred,
                                                    RNA_pred, ligand_pred, max_among_three, pro_minus_max), axis=2)
        print("first 100 in 6 features for model 2: ")
        print("protein")
        print(all_features_model2_3D_np[0:100, :, 0])
        print("DNA")
        print(all_features_model2_3D_np[0:100, :, 1])
        print("RNA")
        print(all_features_model2_3D_np[0:100, :, 2])
        print("ligand")
        print(all_features_model2_3D_np[0:100, :, 3])
        print("max_among_three")
        print(all_features_model2_3D_np[0:100, :, 4])
        print("pro_minus_max")
        print(all_features_model2_3D_np[0:100, :, 5])
        print("label")
        print(label_protein_3D_np[0:100])
        log_time("Start training model2...")
        CV_train_history_model2 = model2.fit(all_features_model2_3D_np,
                                             label_protein_3D_np[train], batch_size=args.batch_size,
                                             epochs=10, verbose=2)
        log_time("Train model2 done.")

        log_time("Start testing on CV-test using model1 and model2 to predict protein binding.")
        print("all_features_3D_np[test].shape: ", all_features_3D_np[test].shape)
        protein_pred, DNA_pred, RNA_pred, ligand_pred = PredictUsingModel1(args, model_protein, model_DNA, model_RNA,
                                                                           model_ligand, all_features_3D_np[test])
        max_among_three, pro_minus_max = ComputeOtherTwoInputsForModel2(protein_pred, DNA_pred, RNA_pred, ligand_pred)
        all_features_model2_3D_np = np.concatenate((protein_pred, DNA_pred,
                                                    RNA_pred, ligand_pred, max_among_three, pro_minus_max), axis=2)
        print("all_features_model2_3D_np.shape: ", all_features_model2_3D_np.shape)
        protein_pred = model2.predict(all_features_model2_3D_np, batch_size=args.batch_size)
        print("label_protein_3D_np[test].shape: ", label_protein_3D_np[test].shape)
        print("protein_pred.shape: ", protein_pred.shape)
        if (int(args.plot_curves) == 1):
            AUC.append(
                PlotRocAndPRCurvesAndMetrics(label_protein_3D_np[test].ravel(), protein_pred.ravel(), args,
                                             "dev_" + str(split_idx) + "_"))

        log_time("Predicting on the test dataset")
        protein_pred, DNA_pred, RNA_pred, ligand_pred = PredictUsingModel1(args, model_protein,
                                                                           model_DNA, model_RNA, model_ligand,
                                                                           all_testing_features_3D_np)
        max_among_three, pro_minus_max = ComputeOtherTwoInputsForModel2(protein_pred, DNA_pred, RNA_pred, ligand_pred)
        all_features_model2_3D_np = np.concatenate((protein_pred, DNA_pred,
                                                    RNA_pred, ligand_pred, max_among_three, pro_minus_max), axis=2)
        print("in testing all_features_model2_3D_np.shape: ", all_features_model2_3D_np.shape)
        protein_pred = model2.predict(all_features_model2_3D_np, batch_size=args.batch_size)
        print("in testing protein_pred.shape: ", protein_pred.shape)
        # y_pred_testing = model_protein.predict(all_testing_features_3D_np, batch_size=args.batch_size).ravel()
        if (int(args.plot_curves) == 1):
            PlotRocAndPRCurvesAndMetrics(label_testing_3D_np.ravel(), protein_pred.ravel(), args, "testing_")
            model_protein.save("savedModel/" + args.prefix + "/" + "model_CV_" + str(split_idx) + ".h5")
        '''
        del model_protein
        # del model_DNA
        # del model_RNA
        # del model_ligand
        # del model2

    # print("AUC[] for each cross validation: ", AUC)
    #
    # with open('csv/result.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     row = ["CV_AUC_mean_STD_" + args.prefix, np.mean(AUC), np.std(AUC)]
    #     writer.writerow(row)
    # csvFile.close()

    # prediction, not in use currently
    if (int(args.perform_prediction) == 1):
        logging.info("train then predict")
        num_of_train = int((1 - args.prediction_split) * all_features_3D_np.shape[0])
        logging.info("num_of_train: %d", num_of_train)
        model_train_history = model_protein.fit(all_features_3D_np[:num_of_train, :, :],
                                                label_protein_3D_np[:num_of_train, :, :],
                                                batch_size=args.batch_size,
                                                epochs=args.epochs, verbose=2, callbacks=[tensorboard])
        if (int(args.plot_curves) == 1):
            PlotAccLossCurves(model_train_history, args)

        protein_pred = model_protein.predict(all_features_3D_np[num_of_train:, :, :],
                                             batch_size=args.batch_size).ravel()
        if (int(args.plot_curves) == 1):
            PlotRocAndPRCurvesAndMetrics(label_protein_3D_np[num_of_train:].ravel(), protein_pred, args,
                                         "evaluation_")

        y_pred_testing = model_protein.predict(all_testing_features_3D_np, batch_size=args.batch_size).ravel()
        if (int(args.plot_curves) == 1):
            PlotRocAndPRCurvesAndMetrics(label_testing_3D_np.ravel(), y_pred_testing, args, "testing_")


# write to csv file csv/results.csv.
def PrintToCSV(prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold):
    header = ['prefix', 'AUC', 'AUPR', 'TP', 'FP', 'TN', 'FN', 'sensitivity', 'specificity', 'recall', 'precision',
              'MCC', 'threshold']
    row = [prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold]
    with open('csv/result.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        # writer.writerow(header)
        writer.writerow(row)
    csvFile.close()


# dirName = subDir + prefix
def MakeDir(subDir, args):
    dirName = subDir + args.prefix
    if not os.path.exists(dirName):
        os.mkdir(dirName)


# Plot training & validation accuracy values
def PlotAccLossCurves(history, args):
    # plot curves
    plt.plot(history.history['acc'])
    if (int(args.cross_validation) == 1):
        plt.plot(history.history['val_acc'
                 ])
    plt.title(args.prefix + ' Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/" + args.prefix + "/accuracy.pdf")
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    if (int(args.cross_validation) == 1):
        plt.plot(history.history['val_loss'])
    plt.title(args.prefix + 'Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/" + args.prefix + "/loss.pdf")
    plt.close()


def GetProgramArguments():
    logging.info("Parsing program arguments...")
    parser = argparse.ArgumentParser(description='BiLSTM_train_pred.py')
    parser._action_groups.pop()

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-fea', '--feature_fn', type=str, required=True, help='str: input feature file name')
    required.add_argument('-lb', '--label_fn', type=str, required=True, help='str: input label file name')
    required.add_argument('-pv', '--pro2vec_fn', type=str, required=True, help='str: pro2vec dictionary file name')

    # optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                          help='float: learning rate')
    optional.add_argument('-do', '--drop_out', type=float, default=0.7,
                          help='float: drop out rate. 1 means no dropout, 0 means drop everything')
    optional.add_argument('-ms', '--model_structure', type=int, default=0,
                          help='int, indicate different models')
    optional.add_argument('-nf', '--num_feature', type=int, default=0,
                          help='int, to use one of all features. [0: all features. 1: ECO. 2: RAA. 3: putativeRSA]')
    optional.add_argument('-ecotr', '--eco_train_fn', type=str, help='training set ECO fn')
    optional.add_argument('-ecote', '--eco_test_fn', type=str, help='testing set ECO fn')
    optional.add_argument('-rsatr', '--rsa_train_dir', type=str, help='the dir that contains training RSA files')
    optional.add_argument('-rsate', '--rsa_test_dir', type=str, help='the dir that contains testing RSA files')
    optional.add_argument('-min', '--mim_seq_len', type=int, default=32,
                          help='int: the min length of each sequence. If a sequence is shorter than that, ignore')
    optional.add_argument('-win', '--window_size', type=int, default=32,
                          help='int: the window size of each split of the sequence')
    optional.add_argument('-vs', '--validation_split', type=float, default=0.25,
                          help='float: percentage of validation split. the last vs percent will be used as validation test data and the (1-vs) percent will be used as training data')
    optional.add_argument('-cv', '--cross_validation', type=int, default=1,
                          help='if cross validation is performed during training')
    optional.add_argument('-gpu', '--use_gpu', type=int, default=0,
                          help='default: use cpu, if specify as 1, use GPU instead')
    # TODO: pd and ps should actually be perform validation
    optional.add_argument('-pd', '--perform_prediction', type=int, default=0,
                          help='if prediction is performed after training')
    optional.add_argument('-ps', '--prediction_split', type=float, default=0.25,
                          help='if prediction is performed after training. The last ps percent will be used as testing data and the (1-ps) percent will be used as training data')
    # TODO: pdfn is the real prediction file name, change the above pd and ps later to vd and pv
    optional.add_argument('-pdfn', '--pred_fn', type=str, help='str: prediction file name in fasta format')
    optional.add_argument('-bs', '--batch_size', type=int, default=1024, help='int: batch size in learning model')
    optional.add_argument('-unit', '--lstm_unit', type=int, default=32, help='int: number of units in LSTM model')
    optional.add_argument('-ep', '--epochs', type=int, default=10, help='int: number of epochs in training')
    optional.add_argument('-pre', '--prefix', type=str, default="default", help='str: prefix on files and plots')
    optional.add_argument('-plot', '--plot_curves', type=int, default=1,
                          help='Plot ROC, PR curves when predition. Plot Accuracy and Loss when validation is preformed.')
    optional.add_argument('-csv', '--print_csv', type=int, default=1,
                          help='Print evaluation metrics into csv files.')
    logging.info("Parsing program arguments done.")
    return parser


def GetRAA(AA, RAA_dict):
    if (AA not in RAA_dict):
        print("[warning]: RAA_dict can't find ", AA, ". Returning 0")
        return 0
    else:
        return RAA_dict[AA]


def CalculateRAAFromASequence(seq, RAA_dict):
    assert (len(seq) >= 2)
    raa = []
    for index, item in enumerate(seq):
        # first letter, 0.5*first + 0.5*second
        if (index == 0):
            # print("first letter: ", item)
            raa.append(0.5 * GetRAA(item, RAA_dict) + 0.5 * GetRAA(seq[index + 1], RAA_dict))
        # last letter, 0.5*last + 0.5*second last
        elif (index == len(seq) - 1):
            # print("last letter: ", item)
            raa.append(0.5 * GetRAA(item, RAA_dict) + 0.5 * GetRAA(seq[index - 1], RAA_dict))
        else:
            # middle ones, 0.5 * i + 0.25 * (i-1) + 0.25 * (i + 1)
            # print("middle letter: ", item)
            raa.append(
                0.5 * GetRAA(item, RAA_dict) + 0.25 * GetRAA(seq[index - 1], RAA_dict) + 0.25 * GetRAA(
                    seq[index + 1],
                    RAA_dict))
    return raa


# dbDir is where RSA database. The DIR of where the RSA files should be loaded, remember to add '/'
def LoadRSAFromPid(pid, seqLength, dbDir):
    RSA_dic = {}
    RSA_dic['A'] = float(121)
    RSA_dic['R'] = float(265)
    RSA_dic['N'] = float(187)
    RSA_dic['D'] = float(187)
    RSA_dic['C'] = float(148)
    RSA_dic['Q'] = float(214)
    RSA_dic['E'] = float(214)
    RSA_dic['G'] = float(97)
    RSA_dic['H'] = float(216)
    RSA_dic['I'] = float(195)
    RSA_dic['L'] = float(191)
    RSA_dic['K'] = float(230)
    RSA_dic['M'] = float(203)
    RSA_dic['F'] = float(228)
    RSA_dic['P'] = float(154)
    RSA_dic['S'] = float(143)
    RSA_dic['T'] = float(163)
    RSA_dic['W'] = float(264)
    RSA_dic['Y'] = float(255)
    RSA_dic['V'] = float(165)

    rsa = []
    # raw_RSA_dir = "../../raw_features/putativeRSA/training_proteins_survey/"
    rsa_fn = dbDir + "asaq." + pid + ".fasta/rasaq.pred"
    try:
        fin = open(rsa_fn, "r")
    except Exception as e:
        print("open file failed. exit now: ", rsa_fn)
        exit(1)

    lines = fin.readlines()
    for x in lines:
        if (x.split(' ')[1] in RSA_dic):
            # rsa.append(float(x.split(' ')[2])/RSA_dic[x.split(' ')[1]])
            rsa.append(float(x.split(' ')[2]))
        else:
            print("[warning]: AA not found in RSA dic ", x.split(' ')[1])
            rsa.append(float(0))

        # for i in range(len(putativeRSA)):
        # if (len(putativeRSA[i]) != len(normalized_RSA[i])):
        #     print("i: ", i)
        #     print("len(putativeRSA[i]): ", len(putativeRSA[i]))
        #     print("len(normalized_RSA[i]): ", len(normalized_RSA[i]))
        #     print("putativeRSA[i]: ", putativeRSA[i])
        #     print("normalized_RSA[i]: ", normalized_RSA[i])
        #     normalized_RSA[i] = [0] * len(putativeRSA[i])
        #     print("after putting zeros, now normalized_RSA[i] is: ", normalized_RSA[i])
    fin.close()
    # some proteins' rsa file fails to be generated, pad with 0
    if (len(rsa) == 0):
        rsa = [0] * seqLength
        print("[warning:]", pid, "has no RSA file. Padding 0 for it.")
    return rsa


def LoadTestingFile(args, Dict_3mer_to_100vec, RAA_dict, max_in_RSA_train, min_in_RSA_train):
    testing_labels_fn = args.pred_fn
    eco_test_fn = args.eco_test_fn
    RAA = []
    RSA = []
    Pro2Vec = []
    label = []
    num_of_pro = 0
    max_pro_len = 0
    min_pro_len = 99999
    total_pro_len = 0

    logging.info("Loading testing file...")
    fin_testing = open(testing_labels_fn, "r")
    while True:
        line_PID = fin_testing.readline()
        line_Seq = fin_testing.readline()
        line_label = fin_testing.readline()
        if not line_label:
            break
        Pro2Vec.append(GetProVecFeature(line_Seq.rstrip('\n').rstrip(' '), Dict_3mer_to_100vec))
        RAA.append(CalculateRAAFromASequence(line_Seq.rstrip('\n').rstrip(' '), RAA_dict))
        RSA.append(LoadRSAFromPid(line_PID.rstrip('\n').rstrip(' ')[1:], len(line_Seq.rstrip('\n').rstrip(' ')),
                                  args.rsa_test_dir))
        label.append(get_array_of_int_from_a_line(line_label))
        max_pro_len = max(max_pro_len, len(line_Seq))
        min_pro_len = min(min_pro_len, len(line_Seq))
        total_pro_len += len(line_Seq)
        num_of_pro += 1
    fin_testing.close()

    logging.info("Loading testing features done.")
    print("[info:]In testing: max_pro_len: ", max_pro_len)
    print("min_pro_len: ", min_pro_len)
    print("avg_pro_len: ", total_pro_len / num_of_pro)
    print("number of proteins in testing: ", num_of_pro)

    ECO = LoadCalculatedECO(eco_test_fn)
    normalized_RSA, max_in_RSA_train, min_in_RSA_train = Normalize2DList(RSA, max_in_RSA_train, min_in_RSA_train)
    return ECO, RAA, normalized_RSA, Pro2Vec, label


def LoadCalculatedECO(ECO_fn, proteins_in_testing=None):
    # load calculated ECO, Saby format
    if (proteins_in_testing is None):
        proteins_in_testing = set()
    ECO_calculated = []  # list of 1D list
    ECO_calculated_fin = open(ECO_fn, "r")
    logging.info("Loading ECO_calculated...")
    while True:
        line_PID = ECO_calculated_fin.readline()
        line_Seq = ECO_calculated_fin.readline()
        line_ECO_calculated = ECO_calculated_fin.readline()
        if not line_ECO_calculated:
            break
        if (line_PID.rstrip('\n').rstrip(' ') not in proteins_in_testing):
            ECO_calculated.append(get_array_of_float_from_a_line(line_ECO_calculated, ','))
    ECO_calculated_fin.close()

    return ECO_calculated


def CountLabelIn2DList(inList):
    num_seq_has_1 = 0
    total_numb_residues = 0
    number_of_1 = 0
    for l in inList:
        total_numb_residues += len(l)
        temp_num_of_1 = l.count(1)
        number_of_1 += temp_num_of_1
        if (temp_num_of_1 != 0):
            num_seq_has_1 += 1
    print("num_seq_has_1: ", num_seq_has_1)
    print("total_numb_residues: ", total_numb_residues)
    print("number_of_1: ", number_of_1)
    print("number_of_1/total_numb_residues: ", number_of_1 / total_numb_residues)


# load three features and labels in HybridNap format.
# load the pro2Vec dictionary and compute on the fly
# If a protein in testing is also in training, it will not use it in training
def LoadFeatureAndLabels(args, Dict_3mer_to_100vec, RAA_dict, proteins_in_testing):
    print("number of proteins in proteins_in_testing: ", len(proteins_in_testing))
    logging.info("Loading features and labels...")
    feature_fn = args.feature_fn
    label_fn = args.label_fn
    ECO_calculated_fn = args.eco_train_fn
    fin_feature = open(feature_fn, "r")
    fin_labels = open(label_fn, "r")

    ECO = []  # list of 1D list
    RAA = []  # list of 1D list; load from Kurgan file
    putativeRSA = []  # list of 1D list
    Pro2Vec = []  # list of 2D numpy array
    label_protein = []  # list of 1D list
    label_DNA = []  # list of 1D list
    label_RNA = []  # list of 1D list
    label_ligand = []  # list of 1D list

    RAA_calculated = []  # list of 1D list; calculated by using RAA_dict
    RSA_calculated = []  # list of 1D list; calculated by loading from ../../raw_features/putativeRSA/
    num_of_pro = 0
    max_pro_len = 0
    min_pro_len = 99999
    total_pro_len = 0

    logging.info("Loading Features...")
    while True:
        line_PID = fin_feature.readline()
        line_Seq = fin_feature.readline()
        line_RAA = fin_feature.readline()
        line_nativeRSA = fin_feature.readline()
        line_putativeRSA = fin_feature.readline()
        line_ECO = fin_feature.readline()
        if not line_ECO:
            break

        # testing proteins can't appear also in training
        if (line_PID.rstrip('\n').rstrip(' ') not in proteins_in_testing):
            Pro2Vec.append(GetProVecFeature(line_Seq.rstrip('\n').rstrip(' '), Dict_3mer_to_100vec))
            ECO.append(get_array_of_float_from_a_line(line_ECO, ' '))
            RAA.append(get_array_of_float_from_a_line(line_RAA, ' '))
            RAA_calculated.append(CalculateRAAFromASequence(line_Seq.rstrip('\n').rstrip(' '), RAA_dict))

            putativeRSA.append(get_array_of_float_from_a_line(line_putativeRSA, ' '))
            RSA_calculated.append(
                LoadRSAFromPid(line_PID.rstrip('\n').rstrip(' ')[1:], len(line_Seq.rstrip('\n').rstrip(' ')),
                               args.rsa_train_dir))
            max_pro_len = max(max_pro_len, len(line_Seq))
            min_pro_len = min(min_pro_len, len(line_Seq))
            total_pro_len += len(line_Seq)
            # if (num_of_pro % 100 == 0):
            #     print("loading the feature of ", num_of_pro, "th protein..")
            num_of_pro += 1
    fin_feature.close()
    logging.info("Loading Features done.")

    print("max_pro_len: ", max_pro_len)
    print("min_pro_len: ", min_pro_len)
    print("avg_pro_len: ", total_pro_len / num_of_pro)

    logging.info("Loading Labels...")
    num_of_pro2 = 0
    while True:
        line_PID = fin_labels.readline()
        line_seq = fin_labels.readline()
        line_seq_PDB = fin_labels.readline()
        line_DNA_binding_label = fin_labels.readline()
        line_RNA_binding_label = fin_labels.readline()
        line_protein_binding_label = fin_labels.readline()
        line_ligand_binding_label = fin_labels.readline()
        if not line_ligand_binding_label:
            break
        if (line_PID.rstrip('\n').rstrip(' ') not in proteins_in_testing):
            label_protein.append(get_array_of_int_from_a_line(line_protein_binding_label))
            label_DNA.append(get_array_of_int_from_a_line(line_DNA_binding_label))
            label_RNA.append(get_array_of_int_from_a_line(line_RNA_binding_label))
            label_ligand.append(get_array_of_int_from_a_line(line_ligand_binding_label))
            num_of_pro2 += 1
    fin_labels.close()
    logging.info("Loading Labels done.")
    # print (label)
    assert (num_of_pro == num_of_pro2)
    print("number of proteins in training that's loaded: ", num_of_pro)

    # load calculated ECO
    ECO_calculated = LoadCalculatedECO(ECO_calculated_fn, proteins_in_testing)

    print("protein stats:")
    CountLabelIn2DList(label_protein)
    print("DNA stats:")
    CountLabelIn2DList(label_DNA)
    print("RNA stats:")
    CountLabelIn2DList(label_RNA)
    print("ligand stats:")
    CountLabelIn2DList(label_ligand)
    return ECO, RAA, putativeRSA, Pro2Vec, label_protein, RAA_calculated, RSA_calculated, ECO_calculated, label_DNA, label_RNA, label_ligand


def BuildSecondModel(args):
    num_of_features = 6  # correspond to the predicted values of protein, DNA, RNA, ligand, max(DNA,RNA,ligand),protein-max(DNA,RNA,ligand))
    logging.info("Building model, the second part...")
    model = Sequential()
    # if (int(args.use_gpu) == 1):
    #     logging.info("Building GPU model2")
    #     model.add(
    #         CuDNNLSTM(args.lstm_unit, return_sequences=True, input_shape=(None, num_of_features)))
    #     # model.add(CuDNNLSTM(args.lstm_unit, return_sequences=True))
    # else:
    #     logging.info("Building CPU model2")
    #     model.add(LSTM(args.lstm_unit, return_sequences=True, input_shape=(None, num_of_features)))
    # model.add(LSTM(args.lstm_unit, return_sequences=True))
    # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(TimeDistributed(Dense(64, activation='sigmoid'), input_shape=(None, num_of_features)))
    model.add(TimeDistributed(Dense(32, activation='sigmoid')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    optimizer_adam = optimizers.Adam(learning_rate=float(args.learning_rate), beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    logging.info("Building the second model done.")
    model.summary()
    return model


def BuildModel(args):
    num_of_features = 1  # regular case: one feature only
    if (args.num_feature == 0 or args.num_feature == -1):
        num_of_features = 3  # 3 features
    elif (args.num_feature == 103 or args.num_feature == 203):
        num_of_features = 103  # 3 features + pro2vec
    elif (args.num_feature == 100):
        num_of_features = 100  # pro2vec

    logging.info("Building model...")
    model = Sequential()
    if (int(args.use_gpu) == 1):
        logging.info("Building GPU model")
        #biLSTM + timeDIstribute Dense 1, default merge_mode
        if (int(args.model_structure) == 0):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True),
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        #biLSTM + timeDIstribute Dense 1, sum merge_mode
        elif (int(args.model_structure) == 1):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True), merge_mode='sum',
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        #biLSTM + timeDIstribute Dense 1 ave merge_mode
        elif (int(args.model_structure) == 2):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True), merge_mode='ave',
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
            # 3: singleLSTM is bad
        elif (int(args.model_structure) == 3):
            model.add(CuDNNLSTM(args.lstm_unit, return_sequences=True, input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        # no TimeDistributed wrapper
        elif (int(args.model_structure) == 4):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True),
                              input_shape=(args.window_size, num_of_features)))
            model.add(Dense(64, activation='sigmoid'))
            model.add(Dense(1, activation='sigmoid'))
            model.add(Reshape((args.window_size,)))
        # no TimeDistributed wrapper, no return sequence
        elif (int(args.model_structure) == 5):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit),
                              input_shape=(args.window_size, num_of_features)))
            model.add(Dense(args.window_size, activation='sigmoid'))
            model.add(Reshape((args.window_size,)))
        elif (int(args.model_structure) == 8):
            model.add(
                Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True),
                              input_shape=(args.window_size, num_of_features)))
            model.add(Dropout(args.drop_out))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))

    else:
        logging.info("Building CPU model")
        if (int(args.model_structure) == 0):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True),
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        elif (int(args.model_structure) == 1):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True), merge_mode='sum',
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        elif (int(args.model_structure) == 2):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True), merge_mode='avg',
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
                #1 with dropout
        elif (int(args.model_structure) == 6):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True, dropout=float(args.drop_out)),
                              input_shape=(args.window_size, num_of_features)))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        #0 with recurrent_dropout
        elif (int(args.model_structure) == 7):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True, recurrent_dropout=float(args.drop_out)),
                              input_shape=(args.window_size, num_of_features)))
            model.add(Dropout(args.drop_out))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
        #0 with only a dropout layer
        elif (int(args.model_structure) == 8):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=True),
                              input_shape=(args.window_size, num_of_features)))
            model.add(Dropout(args.drop_out))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
            model.add(Reshape((args.window_size,)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


    logging.info("Building Model done.")
    # model.summary()
    return model


def log_time(prefix):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    logging.info("%s: %s", prefix, st)


# dim: delimiter
def get_3mer_and_np100vec_from_a_line(line, dim):
    np100 = []
    line = line.rstrip('\n').rstrip(' ').split(dim)
    # print (line)
    three_mer = line.pop(0)
    # print(three_mer)
    # print (line)
    np100 += [float(i) for i in line]
    np100 = np.asarray(np100)
    # print(np100)
    return three_mer, np100


# First load all the 3 grams 1x100 vectors for any given 3-mer
def LoadProtVec3Grams(protVecFN):
    # a dictionary that stores <3mer, vec>
    Dict_3mer_to_100vec = {}
    f = open(protVecFN, "r")
    while True:
        line = f.readline()
        if not line:
            break
        three_mer, np100vec = get_3mer_and_np100vec_from_a_line(line, '\t')
        Dict_3mer_to_100vec[three_mer] = np100vec
    return Dict_3mer_to_100vec


# return the ProVec100 feature values of a given protein sequence. Return a np 2D matrix with size (100 * pro_len )
def GetProVecFeature(pSeq, dic3MerToVec):
    winSize = 9
    # print("winSize/2 = ", int(winSize / 2))
    res = np.zeros((len(pSeq), 100))
    for index, item in enumerate(pSeq):
        # print(index, ": ", item)
        # beginning and end, use <unk>
        if (index < int(winSize / 2) or index >= (len(pSeq) - int(winSize / 2))):
            res[index] = dic3MerToVec["<unk>"]
        # amino acids that in the middle
        else:
            # the feature value become the average of 7 values centered in index
            for i in range(index - int(winSize / 2), index - int(winSize / 2) + winSize - 2):
                # print("i: ", i, "index: ", index, "len(pSeq): ", len(pSeq))
                # print("pSeq[i:i+3]: ", pSeq[i:i + 3])
                if (pSeq[i:i + 3] in dic3MerToVec):
                    res[index] += dic3MerToVec.get(pSeq[i:i + 3])
                else:
                    print("[warning]: dic3MerToVec can't find ", pSeq[i:i + 3], ". Returning <unk>")
                    res[index] += dic3MerToVec.get("<unk>")

            res[index] /= (winSize - 2)
    res = np.fliplr(res)
    res = np.rot90(res)
    return (res)


def BuildRAADictionary():
    RAA_table = np.array(
        [-0.08, 0.12, -0.15, -0.33, 0.76, -0.11, -0.34, -0.25, 0.18, 0.71,
         0.61, -0.38, 0.92, 1.18, -0.17, -0.13, -0.07, 0.95, 0.71, 0.37])
    max_RAA = np.amax(RAA_table)
    min_RAA = np.amin(RAA_table)
    print("max_RAA: ", max_RAA)
    print("min_RAA: ", min_RAA)
    normolized_RAA_table = (RAA_table - min_RAA) / (max_RAA - min_RAA)
    print("normalized_RAA_table: ", normolized_RAA_table)
    # normalized_RAA_table:
    # [0.19230769 0.32051282 0.1474359  0.03205128 0.73076923 0.17307692 0.02564103 0.08333333 0.35897436 0.69871795
    # 0.63461538 0. 0.83333333 1. 0.13461538 0.16025641 0.19871795 0.8525641 0.69871795 0.48076923]

    RAA_dict = {}
    RAA_dict['A'] = normolized_RAA_table[0]  # 0.19230769
    RAA_dict['R'] = normolized_RAA_table[1]  # 0.32051282
    RAA_dict['N'] = normolized_RAA_table[2]  # 0.1474359
    RAA_dict['D'] = normolized_RAA_table[3]  # 0.03205128
    RAA_dict['C'] = normolized_RAA_table[4]  # 0.73076923
    RAA_dict['Q'] = normolized_RAA_table[5]  # 0.17307692
    RAA_dict['E'] = normolized_RAA_table[6]  # 0.02564103
    RAA_dict['G'] = normolized_RAA_table[7]  # 0.08333333
    RAA_dict['H'] = normolized_RAA_table[8]  # 0.35897436
    RAA_dict['I'] = normolized_RAA_table[9]  # 0.69871795
    RAA_dict['L'] = normolized_RAA_table[10]  # 0.63461538
    RAA_dict['K'] = normolized_RAA_table[11]  # 0
    RAA_dict['M'] = normolized_RAA_table[12]  # 0.83333333
    RAA_dict['F'] = normolized_RAA_table[13]  # 1
    RAA_dict['P'] = normolized_RAA_table[14]  # 0.13461538
    RAA_dict['S'] = normolized_RAA_table[15]  # 0.16025641
    RAA_dict['T'] = normolized_RAA_table[16]  # 0.19871795
    RAA_dict['W'] = normolized_RAA_table[17]  # 0.8525641
    RAA_dict['Y'] = normolized_RAA_table[18]  # 0.69871795
    RAA_dict['V'] = normolized_RAA_table[19]  # 0.48076923

    return RAA_dict


# inputs: 2D list of floats
def CheckDiff(list1, list2):
    list1_1D = list(chain.from_iterable(list1))
    list2_1D = list(chain.from_iterable(list2))
    diff = np.array(list1_1D) - np.array(list2_1D)
    diff = np.absolute(diff)
    cnt_big005 = 0
    cnt_big01 = 0
    for i in diff:
        if i > 0.05:
            cnt_big005 += 1
        if i > 0.1:
            cnt_big01 += 1
    print("number of elements that have diff > 0.05 are: ", cnt_big005)
    print("number of elements that have diff > 0.1 are: ", cnt_big01)
    print("mean: ", np.mean(diff))
    print("std: ", np.std(diff))


# if max_in_train and min_in_train are specified, use them; otherwise, detect them
# if in training, return the max and min for testing to use
def Normalize2DList(originalList, max_in_train=-1, min_in_train=-1):
    if (max_in_train == -1 and min_in_train == -1):  # unspecified
        print("[info:] normalizing in training")
        originalListNp = np.array(list(chain.from_iterable(originalList)))
        max = np.amax(originalListNp)
        min = np.amin(originalListNp)
        # print("[Info:] Normalizing max1: ", max, " min: ", min)
    else:
        print("[Info:] normalizing in testing")
        max = max_in_train
        min = min_in_train
    print("[Info:] Normalizing max: ", max, " min: ", min)
    ret = [[(i - min) / (max - min) for i in subList] for subList in originalList]
    # print("originalList: ",originalList)
    # print("normalized list: ", ret)
    return ret, max, min


def CheckTrainAndTestDataSet(args):
    logging.info("Checking the proteins in Training and Testing Dataset")
    proteins_in_training = set()
    train_fn = args.label_fn
    test_fn = args.pred_fn

    fin_train = open(train_fn, "r")
    lines = fin_train.readlines()
    for x in lines:
        if (x[0] == '>'):
            proteins_in_training.add(x.rstrip('\n').rstrip(' '))
    fin_train.close()
    print("number of proteins in training: ", len(proteins_in_training))

    cnt = 0
    proteins_in_testing = set()
    fin_test = open(test_fn, "r")
    lines = fin_test.readlines()
    for x in lines:
        if (x[0] == '>'):
            proteins_in_testing.add(x.rstrip('\n').rstrip(' '))
            if (x.rstrip('\n').rstrip(' ') in proteins_in_training):
                cnt += 1
                print("[warning:]Testing data ", x, "already exists in training set")
    print("number of proteins in testing that exist also in training: ", cnt)
    fin_test.close()
    return proteins_in_testing


def main():
    logging.basicConfig(format='[%(levelname)s] line %(lineno)d: %(message)s', level='INFO')
    print("test print")
    log_time("Program started")
    parser = GetProgramArguments()
    args = parser.parse_args()
    # create directory
    MakeDir('plots/', args)
    MakeDir('logs/', args)
    MakeDir('savedModel/', args)
    proteins_in_testing = CheckTrainAndTestDataSet(args)
    print("program arguments are: ", args)
    RAA_dict = BuildRAADictionary()

    Dict_3mer_to_100vec = LoadProtVec3Grams(args.pro2vec_fn)
    print("Dict_3mer_to_100vec size: ", len(Dict_3mer_to_100vec))
    ECO, RAA, putativeRSA, Pro2Vec, label_protein, RAA_calculated, RSA_calculated, ECO_calculated, label_DNA, label_RNA, label_ligand = \
        LoadFeatureAndLabels(args, Dict_3mer_to_100vec, RAA_dict, proteins_in_testing)
    print("check diff between ECO and ECO_calculated")
    CheckDiff(ECO_calculated, ECO)
    # ECO is naturally normalized, so the next 3 lines are not needed
    # normalized_ECO_calculated = Normalize2DList(ECO_calculated)
    # print("check diff between ECO and normalized_ECO_calculated")
    # CheckDiff(normalized_ECO_calculated, ECO)
    print("check diff of RAA")
    CheckDiff(RAA, RAA_calculated)
    # print("putativeRSA: ")
    # print(putativeRSA)
    # print("RSA_calculated: ")
    # print(RSA_calculated)
    normalized_RSA, max_in_RSA_train, min_in_RSA_train = Normalize2DList(RSA_calculated)
    # print("putativeRSA[0]: ", putativeRSA[0])
    # print("normalized_RSA[0]: ", normalized_RSA[0])
    # print("putativeRSA: ")
    # print(putativeRSA)
    # print("RSA_calculated: ")
    # print(RSA_calculated)
    # print("normalized_RSA: ")
    # print(normalized_RSA)

    print("Some calculated RSA is empty because the sequence is too short. pad 0")
    assert (len(putativeRSA) == len(normalized_RSA))
    for i in range(len(putativeRSA)):
        if (len(putativeRSA[i]) != len(normalized_RSA[i])):
            print("i: ", i)
            print("len(putativeRSA[i]): ", len(putativeRSA[i]))
            print("len(normalized_RSA[i]): ", len(normalized_RSA[i]))
            print("putativeRSA[i]: ", putativeRSA[i])
            print("normalized_RSA[i]: ", normalized_RSA[i])
            normalized_RSA[i] = [0] * len(putativeRSA[i])
            print("after putting zeros, now normalized_RSA[i] is: ", normalized_RSA[i])
    print("check diff of RSA")
    CheckDiff(putativeRSA, normalized_RSA)

    ECO_testing, RAA_testing, RSA_testing, Pro2Vec_testing, label_testing = LoadTestingFile(args,
                                                                                            Dict_3mer_to_100vec,
                                                                                            RAA_dict,
                                                                                            max_in_RSA_train,
                                                                                            min_in_RSA_train)

    # print("pro2vec before split first protein")
    # print(Pro2Vec[0])

    TrainModel(ECO, RAA, putativeRSA, ECO_calculated, RAA_calculated, normalized_RSA, Pro2Vec, label_protein, args,
               ECO_testing, RAA_testing, RSA_testing, Pro2Vec_testing, label_testing, label_DNA, label_RNA,
               label_ligand)

    log_time("Program ended")


if __name__ == '__main__':
    main()
