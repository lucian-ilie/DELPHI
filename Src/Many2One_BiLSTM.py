################################################
# fix the random see value so the results are re-producible
seed_value = 7
import numpy as np
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
import os
import csv
import logging
import datetime
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Flatten, Reshape, TimeDistributed, Bidirectional, CuDNNLSTM, CuDNNGRU, Dropout, Input, Conv2D, MaxPool2D, ConvLSTM2D, SpatialDropout2D, Conv1D, MaxPool1D, Concatenate, BatchNormalization, Activation
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
from keras.utils import plot_model
from itertools import chain
import argparse
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers


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
    F1 = 2 * (precision * recall) / (precision + recall)
    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    # same as sensitivity
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("MCC: ", MCC)
    print("F1: ", F1)
    return TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1


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


# split a 1D list into window size. Shifting step is 1. For the beginning and end of the sequence, pad with 0
def Split1Dlist2NpArrays(args, inputList):
    list_1d_after_split = []
    win_size = args.window_size
    assert (win_size < len(inputList))
    for x in range(len(inputList)):
        sta = (int)(x - (win_size - 1) / 2)
        end = (int)(x + (win_size - 1) / 2)
        if (sta < 0):
            # pad before
            assert (end >= 0)
            list_1d_after_split.extend([0] * abs(sta) + inputList[0:end + 1])
        elif (end >= len(inputList)):
            # pad after
            list_1d_after_split.extend(inputList[sta:len(inputList)] + [0] * (end - len(inputList) + 1))
        else:
            # normal
            list_1d_after_split.extend(inputList[sta:end + 1])
    return list_1d_after_split

# split input 2D list
def Split2DList2NpArrays(args, input2DList):
    # window size has to be odd
    assert (args.window_size % 2 != 0)
    input2DList_after_split = []
    for input1DList in input2DList:
        input1DList_after_split = Split1Dlist2NpArrays(args, input1DList)
        input2DList_after_split.extend(input1DList_after_split)

    np_2d_array = np.asarray(input2DList_after_split)
    np_2d_array = np_2d_array.reshape(-1, args.window_size)
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
    TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1_score = CalculateEvaluationMetrics(truth, pred_binary)
    if (int(args.print_csv) == 1):
        PrintToCSV(csvPre + args.prefix, au_roc, aupr, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC,
                   threshold, F1_score)
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
    # print(np_2d)
    np_3D = np_2d.reshape(np_2d.shape[0], np_2d.shape[1], 1)
    return np_3D

def TrainModel(args, train_all_features_np3D, train_label_np_2D, test_all_features_np3D, test_label_np_2D ):
    log_time("in TrainModel")
    # get the shape of a 1D feature for pro2Vec to use
    # num_row = ECO_3D_np.shape[0]
    # num_col = ECO_3D_np.shape[1]
    # num_row_testing = ECO_testing_3D_np.shape[0]
    # num_col_testing = ECO_testing_3D_np.shape[1]
    #
    # pro2vec_3D_np = SplitPro2Vec2NpArrays(args, pro2vec, num_row, num_col)
    # pro2vec_testing_3D_np = SplitPro2Vec2NpArrays(args, pro2vec_testing, num_row_testing, num_col_testing)


    cur_time = time.time()
    cur_time_formatted = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    tensorboard = TensorBoard(log_dir="tensorboard_log/{}".format(args.prefix))

    # cross validation
    kfold = KFold(n_splits=8, shuffle=False)
    split_idx = 0
    print("train_all_features_np3D.shape: ", train_all_features_np3D.shape)
    print("train_label_np_2D.shape: ", train_label_np_2D.shape)
    print("test_all_features_np3D.shape: ", test_all_features_np3D.shape)
    print("test_label_np_2D.shape: ", test_label_np_2D.shape)
    logging.info("Performirestore_best_weightsng cross validation")
    AUC = []
    for train, test in kfold.split(train_all_features_np3D):
        model = BuildModel(args)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience, restore_best_weights=True)
        # mc = ModelCheckpoint("savedModel/" + args.prefix + "/" + "SavedModel-{epoch:02d}.h5", monitor='val_loss', mode='min',
        #                      verbose=1, save_best_only=False, period=15)
        mc = ModelCheckpoint("savedModel/" + args.prefix + "/" + "SavedModelAndWeights.h5", monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True)
        split_idx += 1
        print("split_idx: ", split_idx)
        if (split_idx > 1):
            break

        print("train: ", train)
        print("test: ", test)
        print("train_feature.shape: ", train_all_features_np3D[train].shape)
        print("test_feature.shape: ", train_all_features_np3D[test].shape)
        print("train_label.shape: ", train_label_np_2D[train].shape)
        print("test_label.shape: ", train_label_np_2D[test].shape)
        logging.info("Start training model...")

        #export training and validation data
        if (args.save_model):
            np.save("savedModel/saved_train_validation_data/train_feature.npy", train_all_features_np3D[train])
            np.save("savedModel/saved_train_validation_data/train_label.npy", train_label_np_2D[train])
            np.save("savedModel/saved_train_validation_data/validation_feature.npy", train_all_features_np3D[test])
            np.save("savedModel/saved_train_validation_data/validation_label.npy", train_label_np_2D[test])

        class_weights_protein = class_weight.compute_class_weight('balanced', np.unique(train_label_np_2D.ravel()),
                                                                  train_label_np_2D.ravel())
        print("class_weights_protein:", class_weights_protein)
        log_time("Start training model1 protein...")
        CV_train_history_protein = model.fit(train_all_features_np3D[train], train_label_np_2D[train],
                                             callbacks=[tensorboard, mc, es], shuffle=True, batch_size=args.batch_size,
                                             class_weight=class_weights_protein, epochs=args.epochs, verbose=2,
                                             validation_data=(train_all_features_np3D[test], train_label_np_2D[test]))

        time_end = time.time()
        time_elapsed = round((time_end - time_start) / 3600, 1)
        ep = es.stopped_epoch

        protein_pred_on_CV_train = model.predict(train_all_features_np3D[train], batch_size=args.batch_size)
        PlotRocAndPRCurvesAndMetrics(train_label_np_2D[train].ravel(), protein_pred_on_CV_train.ravel(), args,
                                     "Train_" + "GPU" + str(args.use_gpu) + "_" + str(time_elapsed) + "hours_" +
                                     str(ep) + "ep_")

        log_time("Start validation...")
        protein_pred_on_CV_test = model.predict(train_all_features_np3D[test], batch_size=args.batch_size)
        PlotRocAndPRCurvesAndMetrics(train_label_np_2D[test].ravel(), protein_pred_on_CV_test.ravel(), args,
                                     "Validation_" + "GPU" + str(args.use_gpu) + "_" + str(time_elapsed) + "hours_" +
                                     str(ep) + "ep_")

        log_time("Start testing...")
        y_pred_testing = model.predict(test_all_features_np3D, batch_size=args.batch_size).ravel()
        if (int(args.plot_curves) == 1):
            PlotRocAndPRCurvesAndMetrics(test_label_np_2D.ravel(), y_pred_testing.ravel(), args, "SCRIBER_test_")
        del model


    # print("AUC[] for each cross validation: ", AUC)
    #
    # with open('csv/result.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     row = ["CV_AUC_mean_STD_" + args.prefix, np.mean(AUC), np.std(AUC)]
    #     writer.writerow(row)
    # csvFile.close()

# write to csv file csv/results.csv.
def PrintToCSV(prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold, F1_score):
    header = ['prefix', 'AUC', 'AUPR', 'TP', 'FP', 'TN', 'FN', 'sensitivity', 'specificity', 'recall', 'precision',
              'MCC', 'threshold', 'F1_score']
    row = [prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold, F1_score]
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
    required.add_argument('-trdb', '--training_dataset_prefix', type=str, required=True,
                          help='str: training dataset prefix, currently DS_72","DS_164","DS_186","SCRIBER_test","SCRIBER_train","survey_test","survey_train"')
    required.add_argument('-tedb', '--testing_dataset_prefix', type=str, required=True,
                          help='str: testing dataset prefix, currently DS_72","DS_164","DS_186","SCRIBER_test","SCRIBER_train","survey_test","survey_train"')

    required.add_argument('-trPro', '--training_protein_fn', type=str, required=True,
                          help='str: proteins used for training in id_seq_label format')
    required.add_argument('-tePro', '--testing_protein_fn', type=str, required=True,
                          help='str: proteins used for testing in id_seq_label format')
    required.add_argument('-pv', '--pro2vec_fn', type=str, required=True, help='str: pro2vec dictionary file name')

    # optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                          help='float: learning rate')
    optional.add_argument('-do', '--drop_out', type=float, default=0.7,
                          help='float: drop out rate. 1 means no dropout, 0 means drop everything')
    optional.add_argument('-ms', '--model_structure', type=int, default=0,
                          help='int, indicate different models')
    optional.add_argument('-pa', '--patience', type=int, default=5,
                          help='int, patience in early stopping')
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
    optional.add_argument('-sm', '--save_model', type=int, default=0,
                          help='if np file needs to be saved')
    optional.add_argument('-ld', '--load_data', type=int, default=0,
                          help='if loads saved np files directly')
    optional.add_argument('-ps', '--prediction_split', type=float, default=0.25,
                          help='if prediction is performed after training. The last ps percent will be used as testing data and the (1-ps) percent will be used as training data')
    # TODO: pdfn is the real prediction file name, change the above pd and ps later to vd and pv
    optional.add_argument('-pdfn', '--pred_fn', type=str, help='str: prediction file name in fasta format')
    optional.add_argument('-bs', '--batch_size', type=int, default=1024, help='int: batch size in learning model')
    optional.add_argument('-unit', '--lstm_unit', type=int, default=32, help='int: number of units in LSTM model')
    optional.add_argument('-ks', '--kernel_size', type=int, default=7, help='int: kernel size in convolution')
    optional.add_argument('-fil', '--filter_size', type=int, default=16, help='int: filter size in convolution')
    optional.add_argument('-ep', '--epochs', type=int, default=10, help='int: number of epochs in training')
    optional.add_argument('-pre', '--prefix', type=str, default="default", help='str: prefix on files and plots')
    optional.add_argument('-plot', '--plot_curves', type=int, default=1,
                          help='Plot ROC, PR curves when predition. Plot Accuracy and Loss when validation is preformed.')
    optional.add_argument('-csv', '--print_csv', type=int, default=1,
                          help='Print evaluation metrics into csv files.')
    logging.info("Parsing program arguments done.")
    return parser

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

def BuildModel(args):
    num_feature = 1
    if ((int)(args.num_feature) == 0):
        num_feature = 118
    if ((int)(args.num_feature) == -1):
        num_feature = 38
    elif ((int)(args.num_feature) == 5):
        num_feature = 100

    logging.info("Building model...")
    model = Sequential()
    if (int(args.use_gpu) == 1):
        logging.info("Building GPU model")
        # Dense only
        if (int(args.model_structure) == 0):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Dense(64, activation="sigmoid")(input_features)
            out = Dense(32, activation="sigmoid")(out)
            out = Flatten()(out)
            out = Dense(1, activation="sigmoid")(out)
            model = Model(inputs = input_features, outputs = out)
        # BiLSTM no dropout
        elif (int(args.model_structure) == 1):
            model.add(Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=False), input_shape=((int)(args.window_size), num_feature)))
            model.add(Dense(1, activation='sigmoid'))
        # BiLSTM + LSTM no dropout
        elif (int(args.model_structure) == 2):
            model.add(Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True), input_shape=((int)(args.window_size), num_feature)))
            model.add(CuDNNLSTM(args.lstm_unit, return_sequences=False))
            model.add(Dense(1, activation='sigmoid'))
        # BiLSTM + LSTM + dropout
        elif (int(args.model_structure) == 3):
            model.add(Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True), input_shape=((int)(args.window_size), num_feature)))
            model.add(CuDNNLSTM(args.lstm_unit, return_sequences=False))
            model.add(Dropout(args.drop_out))
            model.add(Dense(1, activation='sigmoid'))
        # BiLSTM + LSTM + L2 regularizer
        elif (int(args.model_structure) == 4):
            model.add(Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)), input_shape=((int)(args.window_size), num_feature)))
            model.add(CuDNNLSTM(args.lstm_unit, return_sequences=False))
            model.add(Dense(1, activation='sigmoid'))
        # BiLSTM + LSTM + L2 regularizer on recurrent_regularizer
        elif (int(args.model_structure) == 5):
            model.add(Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)), input_shape=((int)(args.window_size), num_feature)))
            model.add(CuDNNLSTM(args.lstm_unit, return_sequences=False))
            model.add(Dense(1, activation='sigmoid'))
        # GRU + L2 regularizer on recurrent_regularizer, pretty good. ep 15 makes ds186 and ds164 good
        elif (int(args.model_structure) == 6):
            model.add(Bidirectional(
                CuDNNGRU(units=args.lstm_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)),
                input_shape=((int)(args.window_size), num_feature)))
            model.add(CuDNNGRU(units=args.lstm_unit, return_sequences=False))
            model.add(Dense(64, activation='sigmoid'))
            model.add(Dense(1, activation='sigmoid'))
        # GRU. pretty good. ep 15 makes ds72 and ds164 and ds 186good
        elif (int(args.model_structure) == 7):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Bidirectional(CuDNNGRU(name="gru_right", units=args.lstm_unit, return_sequences=True), name="bidirectional_right")(input_features)
            out = Dropout(args.drop_out)(out)
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid', name="dense_RNN_1")(out)
            out = Dropout(args.drop_out)(out)
            out = Dense(1, activation='sigmoid', name="dense_RNN_2")(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D
        elif (int(args.model_structure) == 8):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", activation="relu")(out)
            # model.add(SpatialDropout2D(0.5))
            # out = Conv2D(filters= 1, kernel_size=5, padding="same", activation="relu")(out)
            # out = Conv2D(filters= 1, kernel_size=7, padding="same", activation="relu")(out)
            out = MaxPool2D(pool_size=3)(out)
            # model.add(Dropout(0.5))
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            # model.add(Dropout(0.5))
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D with kernel size rectangular
        elif (int(args.model_structure) == 81):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=(args.kernel_size, num_feature), data_format="channels_last", padding="same", activation="relu")(out)
            out = MaxPool2D(pool_size=3)(out)
            # model.add(Dropout(0.5))
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            # model.add(Dropout(0.5))
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D x 2
        elif (int(args.model_structure) == 82):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", activation="relu")(out)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", activation="relu")(out)
            out = MaxPool2D(pool_size=3)(out)
            # model.add(Dropout(0.5))
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            # model.add(Dropout(0.5))
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D with batch norm
        elif (int(args.model_structure) == 83):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", use_bias=False)(out)
            out = BatchNormalization(axis=-1)(out)
            out = MaxPool2D(pool_size=3)(out)
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D with SpatialDropout2D
        elif (int(args.model_structure) == 84):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", activation="relu")(out)
            out = SpatialDropout2D(args.drop_out)(out)
            out = MaxPool2D(pool_size=3)(out)
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv2D with dropout
        elif (int(args.model_structure) == 85):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Reshape((args.window_size, num_feature, 1))(input_features)
            out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last", padding="same", activation="relu", name="conv2d_left")(out)
            out = Dropout(args.drop_out)(out)
            out = MaxPool2D(pool_size=3)(out)
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid', name="dense_CNN_1")(out)
            out = Dropout(args.drop_out)(out)
            out = Dense(1, activation='sigmoid', name="dense_CNN_2")(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv1D
        elif (int(args.model_structure) == 9):
            input_features = Input(shape = ((int)(args.window_size), num_feature))
            out = Conv1D(filters= args.filter_size, kernel_size=args.kernel_size, padding="same")(input_features)
            # model.add(SpatialDropout2D(0.5))
            # out = Conv1D(filters= 8, kernel_size=5, padding="same", activation="relu")(out)
            # out = Conv1D(filters= 1, kernel_size=7, padding="same", activation="relu")(out)
            out = MaxPool1D(pool_size=3)(out)
            # model.add(Dropout(0.5))
            out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            # model.add(Dropout(0.5))
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs = input_features, outputs = out)
        # Conv1D + LSTM, serial
        elif (int(args.model_structure) == 10):
            input_features = Input(shape=((int)(args.window_size), num_feature))
            out = Conv1D(filters=args.filter_size, kernel_size=args.kernel_size, padding="same", activation="relu")(
                input_features)
            # model.add(SpatialDropout2D(0.5))
            # out = Conv2D(filters= 1, kernel_size=5, padding="same", activation="relu")(out)
            # out = Conv2D(filters= 1, kernel_size=7, padding="same", activation="relu")(out)
            out = MaxPool1D(pool_size=3)(out)
            out = Bidirectional(
                CuDNNLSTM(args.lstm_unit, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(out)
            out = CuDNNLSTM(args.lstm_unit, return_sequences=False)(out)
            # model.add(Dropout(0.5))
            # out = Flatten()(out)
            out = Dense(64, activation='sigmoid')(out)
            # model.add(Dropout(0.5))
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs=input_features, outputs=out)
            
        # Conv2D + GRU, assembled model, fine-tune
        elif (int(args.model_structure) == 11):
            input_features = Input(shape=((int)(args.window_size), num_feature))
            # left
            out1 = Reshape((args.window_size, num_feature, 1))(input_features)
            out1 = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, name="conv2d_left", trainable=False,
                          padding="same", data_format="channels_last", activation="relu")(out1)
            out1 = MaxPool2D(pool_size=3)(out1)
            out1 = Flatten()(out1)
            out1 = Dropout(args.drop_out)(out1)

            # out1 = Activation(activation="sigmoid")(out1)
            # out1 = Dropout(args.drop_out)(out1)

            # out1 = Dense(64, activation='sigmoid', name="dense_CNN_1", trainable=False)(out1)
            # out1 = Dropout(args.drop_out)(out1)
            # out1 = Dense(1, activation='sigmoid', name="dense_CNN_2", trainable=False,)(out1)
            # right
            out2 = Bidirectional(CuDNNGRU(args.lstm_unit, return_sequences=True, name="gru_right", trainable=False),
                                 name="bidirectional_right", trainable=False)(input_features)
            out2 = Dropout(args.drop_out)(out2)
            out2 = Flatten()(out2)
            # out2 = Activation(activation="sigmoid")(out2)
            # out2 = Dropout(args.drop_out)(out2)
            # out2 = Dense(64, activation='sigmoid', name="dense_RNN_1", trainable=False)(out2)
            # out2 = Dropout(args.drop_out)(out2)
            # out2 = Dense(1, activation='sigmoid', name="dense_RNN_2", trainable=False,)(out2)
            # combine
            concatenated = Concatenate()([out1, out2])
            out = Dropout(args.drop_out)(concatenated)
            out = Dense(64, activation='sigmoid', name="new_dense1")(out)
            out = Dropout(args.drop_out)(out)
            out = Dense(1, activation='sigmoid', name="new_dense2")(out)
            logging.info("loading weight CNN")
            model = Model(inputs=input_features, outputs=out)
            model.load_weights("/home/j00492398/test_joey/interface-pred/Src/savedModel/bestModels/CNN_best_model_and_weights.h5", by_name=True)
            logging.info("loading weight RNN")
            model.load_weights("/home/j00492398/test_joey/interface-pred/Src/savedModel/bestModels/RNN_best_model_and_weights.h5", by_name=True)

        #Conv2D + LSTM, parallel, pick the good score
        elif (int(args.model_structure) == 12):
            input_features = Input(shape=((int)(args.window_size), num_feature))
            # left
            out1 = Reshape((args.window_size, num_feature, 1))(input_features)
            out1 = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, padding="same", data_format="channels_last", activation="relu")(
                out1)
            out1 = MaxPool2D(pool_size=3)(out1)
            out1 = Flatten()(out1)
            out1 = Dense(64, activation='sigmoid')(out1)
            out1 = Dense(1, activation='sigmoid')(out1)
            # right
            out2 = Bidirectional(
                CuDNNLSTM(args.lstm_unit, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(
                input_features)
            out2 = CuDNNLSTM(args.lstm_unit, return_sequences=False)(out2)
            out2 = Dense(1, activation='sigmoid')(out2)
            # combine
            concatenated = Concatenate()([out1, out2])
            out = Dense(64, activation='sigmoid')(concatenated)
            out = Dense(1, activation='sigmoid')(out)
            model = Model(inputs=input_features, outputs=out)
        #Conv2D + GRU, parallel, train together
        elif (int(args.model_structure) == 13):
            input_features = Input(shape=((int)(args.window_size), num_feature))
            # left
            out1 = Reshape((args.window_size, num_feature, 1))(input_features)
            out1 = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, name="conv2d_left",
                          padding="same", data_format="channels_last", activation="sigmoid")(out1)
            out1 = BatchNormalization(axis=3)(out1)
            # out1 = Dropout(args.drop_out)(out1)
            out1 = MaxPool2D(pool_size=3)(out1)
            out1 = Flatten()(out1)
            # right
            out2 = Bidirectional(CuDNNLSTM(args.lstm_unit, return_sequences=True, name="gru_right"), name="bidirectional_right")(input_features)
            out2 = BatchNormalization()(out2)
            out2 = CuDNNLSTM(args.lstm_unit, return_sequences=True)(out2)
            out2 = BatchNormalization()(out2)
            # out2 = Dropout(args.drop_out)(out2)
            out2 = Flatten()(out2)
            # out2 = Activation(activation="sigmoid")(out2)
            # combine
            concatenated = Concatenate()([out1, out2])
            # out = Dropout(args.drop_out)(concatenated)
            out = Dense(64, activation='sigmoid', name="new_dense1")(concatenated)
            out = BatchNormalization()(out)
            # out = Dropout(args.drop_out)(out)
            out = Dense(1, activation='sigmoid', name="new_dense2")(out)
            model = Model(inputs=input_features, outputs=out)

        else:
            logging.error("model_structure error")
            exit(1)
    else:
        logging.info("Building CPU model")
        if (int(args.model_structure) == 0):
            model.add(
                Bidirectional(LSTM(args.lstm_unit, return_sequences=False), input_shape=((int)(args.window_size), num_feature)))
            model.add(Dropout(args.drop_out))
            model.add(Dense(1, activation='sigmoid'))
        else:
            logging.error("model_structure error")
            exit(1)

    optimizer_adam = optimizers.Adam(lr=(float)(args.learning_rate), beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['acc'])
    logging.info("Building Model done.")
    model.summary()
    plot_model(model, to_file="savedModel/" + args.prefix + "/model_plot.png", show_shapes=True)
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
    f = open(protVecFN, "r")
    while True:
        line = f.readline()
        if not line:
            break
        three_mer, np100vec = get_3mer_and_np100vec_from_a_line(line, '\t')
        Dict_3mer_to_100vec[three_mer] = np100vec.reshape(1, 1, np100vec.shape[0])

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


# load input_fn and write to dict
def Read1DFeature(input_fn, dict):
    fin = open(input_fn, "r")
    while True:
        line_PID = fin.readline().rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        line_feature = fin.readline().rstrip('\n').rstrip(' ')
        if not line_feature:
            break
        dict[line_PID] = get_array_of_float_from_a_line(line_feature, ',')

# for a given sequence, return its pro2vec np 3d array of shape (1, pro_length, 100)
def PreparePro2Vec(seq):
    res_3d_np = np.zeros(shape=(1,len(seq),100))
    for sta in range(len(seq)):
        if (sta == 0):
            # add NA at the beginning
            res_3d_np[:,sta,:] = Dict_3mer_to_100vec["<unk>"]
        elif (sta == len(seq) - 1):
            # add NA at the end
            res_3d_np[:,sta,:] = Dict_3mer_to_100vec["<unk>"]
        else:
            # read 3 mer
            temp_3mer = seq[sta - 1:sta + 2]
            if (temp_3mer not in Dict_3mer_to_100vec):
                print("[warning:] ", temp_3mer , " is not in Dict_3mer_to_100vec. using ,unk instead")
                res_3d_np[:,sta,:] = Dict_3mer_to_100vec["<unk>"]
            else:
                res_3d_np[:,sta,:] = Dict_3mer_to_100vec[temp_3mer]
    return res_3d_np

# load input_fn and write to dict
def ReadNDFeature(input_fasta, dict):
    fin = open(input_fasta, "r")
    while True:
        line_PID = fin.readline().rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        line_Label = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break
        dict[line_PID] = PreparePro2Vec(line_Pseq)
    fin.close()

# load each feature dictionary
def LoadFeatures(args):
    log_time("Loading feature Pro2Vec")
    LoadProtVec3Grams(args.pro2vec_fn)
    ReadNDFeature(args.training_protein_fn, Pro2Vec_train_dic)
    ReadNDFeature(args.testing_protein_fn, Pro2Vec_test_dic)

    log_time("Loading feature ECO")
    Read1DFeature(ECO_DB[args.training_dataset_prefix], ECO_train_dic)
    Read1DFeature(ECO_DB[args.testing_dataset_prefix], ECO_test_dic)
    log_time("Loading feature RAA")
    Read1DFeature(RAA_DB[args.training_dataset_prefix], RAA_train_dic)
    Read1DFeature(RAA_DB[args.testing_dataset_prefix], RAA_test_dic)
    log_time("Loading feature RSA")
    Read1DFeature(RSA_DB[args.training_dataset_prefix], RSA_train_dic)
    Read1DFeature(RSA_DB[args.testing_dataset_prefix], RSA_test_dic)
    log_time("Loading feature Anchor")
    Read1DFeature(Anchor_DB[args.training_dataset_prefix], Anchor_train_dic)
    Read1DFeature(Anchor_DB[args.testing_dataset_prefix], Anchor_test_dic)
    log_time("Loading feature HYD")
    Read1DFeature(HYD_DB[args.training_dataset_prefix], HYD_train_dic)
    Read1DFeature(HYD_DB[args.testing_dataset_prefix], HYD_test_dic)
    log_time("Loading feature PKA")
    Read1DFeature(PKA_DB[args.training_dataset_prefix], PKA_train_dic)
    Read1DFeature(PKA_DB[args.testing_dataset_prefix], PKA_test_dic)
    log_time("Loading feature Pro2Vec_1D")
    Read1DFeature(Pro2Vec_1D_DB[args.training_dataset_prefix], Pro2Vec_1D_train_dic)
    Read1DFeature(Pro2Vec_1D_DB[args.testing_dataset_prefix], Pro2Vec_1D_test_dic)
    log_time("Loading feature HSP")
    Read1DFeature(HSP_DB[args.training_dataset_prefix], HSP_train_dic)
    Read1DFeature(HSP_DB[args.testing_dataset_prefix], HSP_test_dic)

    log_time("Loading feature PHY_Char")
    Read1DFeature(PHY_Char_DB_1[args.training_dataset_prefix], PHY_Char_train_dic_1)
    Read1DFeature(PHY_Char_DB_1[args.testing_dataset_prefix], PHY_Char_test_dic_1)
    Read1DFeature(PHY_Char_DB_2[args.training_dataset_prefix], PHY_Char_train_dic_2)
    Read1DFeature(PHY_Char_DB_2[args.testing_dataset_prefix], PHY_Char_test_dic_2)
    Read1DFeature(PHY_Char_DB_3[args.training_dataset_prefix], PHY_Char_train_dic_3)
    Read1DFeature(PHY_Char_DB_3[args.testing_dataset_prefix], PHY_Char_test_dic_3)

    log_time("Loading feature PHY_Prop")
    Read1DFeature(PHY_Prop_DB_1[args.training_dataset_prefix], PHY_Prop_train_dic_1)
    Read1DFeature(PHY_Prop_DB_1[args.testing_dataset_prefix], PHY_Prop_test_dic_1)
    Read1DFeature(PHY_Prop_DB_2[args.training_dataset_prefix], PHY_Prop_train_dic_2)
    Read1DFeature(PHY_Prop_DB_2[args.testing_dataset_prefix], PHY_Prop_test_dic_2)
    Read1DFeature(PHY_Prop_DB_3[args.training_dataset_prefix], PHY_Prop_train_dic_3)
    Read1DFeature(PHY_Prop_DB_3[args.testing_dataset_prefix], PHY_Prop_test_dic_3)
    Read1DFeature(PHY_Prop_DB_4[args.training_dataset_prefix], PHY_Prop_train_dic_4)
    Read1DFeature(PHY_Prop_DB_4[args.testing_dataset_prefix], PHY_Prop_test_dic_4)
    Read1DFeature(PHY_Prop_DB_5[args.training_dataset_prefix], PHY_Prop_train_dic_5)
    Read1DFeature(PHY_Prop_DB_5[args.testing_dataset_prefix], PHY_Prop_test_dic_5)
    Read1DFeature(PHY_Prop_DB_6[args.training_dataset_prefix], PHY_Prop_train_dic_6)
    Read1DFeature(PHY_Prop_DB_6[args.testing_dataset_prefix], PHY_Prop_test_dic_6)
    Read1DFeature(PHY_Prop_DB_7[args.training_dataset_prefix], PHY_Prop_train_dic_7)
    Read1DFeature(PHY_Prop_DB_7[args.testing_dataset_prefix], PHY_Prop_test_dic_7)

    log_time("Loading feature PSSM")
    Read1DFeature(PSSM_DB_1[args.training_dataset_prefix], PSSM_train_dic_1)
    Read1DFeature(PSSM_DB_2[args.training_dataset_prefix], PSSM_train_dic_2)
    Read1DFeature(PSSM_DB_3[args.training_dataset_prefix], PSSM_train_dic_3)
    Read1DFeature(PSSM_DB_4[args.training_dataset_prefix], PSSM_train_dic_4)
    Read1DFeature(PSSM_DB_5[args.training_dataset_prefix], PSSM_train_dic_5)
    Read1DFeature(PSSM_DB_6[args.training_dataset_prefix], PSSM_train_dic_6)
    Read1DFeature(PSSM_DB_7[args.training_dataset_prefix], PSSM_train_dic_7)
    Read1DFeature(PSSM_DB_8[args.training_dataset_prefix], PSSM_train_dic_8)
    Read1DFeature(PSSM_DB_9[args.training_dataset_prefix], PSSM_train_dic_9)
    Read1DFeature(PSSM_DB_10[args.training_dataset_prefix], PSSM_train_dic_10)
    Read1DFeature(PSSM_DB_11[args.training_dataset_prefix], PSSM_train_dic_11)
    Read1DFeature(PSSM_DB_12[args.training_dataset_prefix], PSSM_train_dic_12)
    Read1DFeature(PSSM_DB_13[args.training_dataset_prefix], PSSM_train_dic_13)
    Read1DFeature(PSSM_DB_14[args.training_dataset_prefix], PSSM_train_dic_14)
    Read1DFeature(PSSM_DB_15[args.training_dataset_prefix], PSSM_train_dic_15)
    Read1DFeature(PSSM_DB_16[args.training_dataset_prefix], PSSM_train_dic_16)
    Read1DFeature(PSSM_DB_17[args.training_dataset_prefix], PSSM_train_dic_17)
    Read1DFeature(PSSM_DB_18[args.training_dataset_prefix], PSSM_train_dic_18)
    Read1DFeature(PSSM_DB_19[args.training_dataset_prefix], PSSM_train_dic_19)
    Read1DFeature(PSSM_DB_20[args.training_dataset_prefix], PSSM_train_dic_20)
    Read1DFeature(PSSM_DB_1[args.testing_dataset_prefix], PSSM_test_dic_1)
    Read1DFeature(PSSM_DB_2[args.testing_dataset_prefix], PSSM_test_dic_2)
    Read1DFeature(PSSM_DB_3[args.testing_dataset_prefix], PSSM_test_dic_3)
    Read1DFeature(PSSM_DB_4[args.testing_dataset_prefix], PSSM_test_dic_4)
    Read1DFeature(PSSM_DB_5[args.testing_dataset_prefix], PSSM_test_dic_5)
    Read1DFeature(PSSM_DB_6[args.testing_dataset_prefix], PSSM_test_dic_6)
    Read1DFeature(PSSM_DB_7[args.testing_dataset_prefix], PSSM_test_dic_7)
    Read1DFeature(PSSM_DB_8[args.testing_dataset_prefix], PSSM_test_dic_8)
    Read1DFeature(PSSM_DB_9[args.testing_dataset_prefix], PSSM_test_dic_9)
    Read1DFeature(PSSM_DB_10[args.testing_dataset_prefix], PSSM_test_dic_10)
    Read1DFeature(PSSM_DB_11[args.testing_dataset_prefix], PSSM_test_dic_11)
    Read1DFeature(PSSM_DB_12[args.testing_dataset_prefix], PSSM_test_dic_12)
    Read1DFeature(PSSM_DB_13[args.testing_dataset_prefix], PSSM_test_dic_13)
    Read1DFeature(PSSM_DB_14[args.testing_dataset_prefix], PSSM_test_dic_14)
    Read1DFeature(PSSM_DB_15[args.testing_dataset_prefix], PSSM_test_dic_15)
    Read1DFeature(PSSM_DB_16[args.testing_dataset_prefix], PSSM_test_dic_16)
    Read1DFeature(PSSM_DB_17[args.testing_dataset_prefix], PSSM_test_dic_17)
    Read1DFeature(PSSM_DB_18[args.testing_dataset_prefix], PSSM_test_dic_18)
    Read1DFeature(PSSM_DB_19[args.testing_dataset_prefix], PSSM_test_dic_19)
    Read1DFeature(PSSM_DB_20[args.testing_dataset_prefix], PSSM_test_dic_20)

    log_time("Loading features done")

# for a given np3D of shape (1,pro_len,100) return a list of 3D np of shape (xx, win_size, 100)
def SplitPro2Vec(np3D_pro2vec, args):
    win_size = args.window_size
    res_list = []

    for x in range(np3D_pro2vec.shape[1]):
        sta = (int)(x - (win_size - 1) / 2)
        end = (int)(x + (win_size - 1) / 2)
        if (sta < 0):
            # pad before
            assert (end >= 0)
            np_temp_beginning = np.repeat(Dict_3mer_to_100vec.get("<unk>"), abs(sta), axis=1)
            res_list.extend(np.concatenate((np_temp_beginning, np3D_pro2vec[:, 0:end+1, :]), axis=1))

        elif (end >= np3D_pro2vec.shape[1]):
            # pad after
            np_temp_ending = np.repeat(Dict_3mer_to_100vec.get("<unk>"), (end - np3D_pro2vec.shape[1] + 1), axis=1)
            res_list.extend(np.concatenate((np3D_pro2vec[:,sta:np3D_pro2vec.shape[1],:],np_temp_ending), axis = 1))
        else:
            # normal
            res_list.extend(np3D_pro2vec[:,sta:end+1,:])
    return res_list

# input_fn: a file in pid, pSeq, label format
# output: 1. np_3D of all features; 2. np_2D of label
def LoadLabelsAndFormatFeatures(args, input_fn, isTrain):
    ECO_2DList = []
    RAA_2DList = []
    RSA_2DList = []
    Pro2Vec_1D_2DList = []
    Anchor_2DList = []
    HSP_2DList = []
    HYD_2DList = []
    PKA_2DList = []
    PHY_Char_2DList_1 = []
    PHY_Char_2DList_2 = []
    PHY_Char_2DList_3 = []
    PHY_Prop_2DList_1 = []
    PHY_Prop_2DList_2 = []
    PHY_Prop_2DList_3 = []
    PHY_Prop_2DList_4 = []
    PHY_Prop_2DList_5 = []
    PHY_Prop_2DList_6 = []
    PHY_Prop_2DList_7 = []

    PSSM_2DList_1 = []
    PSSM_2DList_2 = []
    PSSM_2DList_3 = []
    PSSM_2DList_4 = []
    PSSM_2DList_5 = []
    PSSM_2DList_6 = []
    PSSM_2DList_7 = []
    PSSM_2DList_8 = []
    PSSM_2DList_9 = []
    PSSM_2DList_10 = []
    PSSM_2DList_11 = []
    PSSM_2DList_12 = []
    PSSM_2DList_13 = []
    PSSM_2DList_14 = []
    PSSM_2DList_15 = []
    PSSM_2DList_16 = []
    PSSM_2DList_17 = []
    PSSM_2DList_18 = []
    PSSM_2DList_19 = []
    PSSM_2DList_20 = []

    Pro2Vec_List = []
    # all labels put together
    label_1DList = []
    fin = open(input_fn, "r")
    while True:
        line_PID = fin.readline().rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        line_label = fin.readline().rstrip('\n').rstrip(' ')
        if not line_label:
            break
        if (len(line_Pseq) < (int)(args.mim_seq_len)):
            continue
        list1D_label_this_line = get_array_of_int_from_a_line(line_label)
        if (isTrain):
            list1D_ECO = ECO_train_dic[line_PID]
            list1D_RAA = RAA_train_dic[line_PID]
            list1D_RSA = RSA_train_dic[line_PID]
            list1D_Pro2Vec_1D = Pro2Vec_1D_train_dic[line_PID]
            list1D_Anchor = Anchor_train_dic[line_PID]
            list1D_HSP = HSP_train_dic[line_PID]
            list1D_HYD = HYD_train_dic[line_PID]
            list1D_PKA = PKA_train_dic[line_PID]
            list1D_PHY_Char_1 = PHY_Char_train_dic_1[line_PID]
            list1D_PHY_Char_2 = PHY_Char_train_dic_2[line_PID]
            list1D_PHY_Char_3 = PHY_Char_train_dic_3[line_PID]
            list1D_PHY_Prop_1 = PHY_Prop_train_dic_1[line_PID]
            list1D_PHY_Prop_2 = PHY_Prop_train_dic_2[line_PID]
            list1D_PHY_Prop_3 = PHY_Prop_train_dic_3[line_PID]
            list1D_PHY_Prop_4 = PHY_Prop_train_dic_4[line_PID]
            list1D_PHY_Prop_5 = PHY_Prop_train_dic_5[line_PID]
            list1D_PHY_Prop_6 = PHY_Prop_train_dic_6[line_PID]
            list1D_PHY_Prop_7 = PHY_Prop_train_dic_7[line_PID]
            list1D_PSSM_1 = PSSM_train_dic_1[line_PID]
            list1D_PSSM_2 = PSSM_train_dic_2[line_PID]
            list1D_PSSM_3 = PSSM_train_dic_3[line_PID]
            list1D_PSSM_4 = PSSM_train_dic_4[line_PID]
            list1D_PSSM_5 = PSSM_train_dic_5[line_PID]
            list1D_PSSM_6 = PSSM_train_dic_6[line_PID]
            list1D_PSSM_7 = PSSM_train_dic_7[line_PID]
            list1D_PSSM_8 = PSSM_train_dic_8[line_PID]
            list1D_PSSM_9 = PSSM_train_dic_9[line_PID]
            list1D_PSSM_10 = PSSM_train_dic_10[line_PID]
            list1D_PSSM_11 = PSSM_train_dic_11[line_PID]
            list1D_PSSM_12 = PSSM_train_dic_12[line_PID]
            list1D_PSSM_13 = PSSM_train_dic_13[line_PID]
            list1D_PSSM_14 = PSSM_train_dic_14[line_PID]
            list1D_PSSM_15 = PSSM_train_dic_15[line_PID]
            list1D_PSSM_16 = PSSM_train_dic_16[line_PID]
            list1D_PSSM_17 = PSSM_train_dic_17[line_PID]
            list1D_PSSM_18 = PSSM_train_dic_18[line_PID]
            list1D_PSSM_19 = PSSM_train_dic_19[line_PID]
            list1D_PSSM_20 = PSSM_train_dic_20[line_PID]

            np3D_pro2vec = Pro2Vec_train_dic[line_PID]
        else:  # isTesting
            list1D_ECO = ECO_test_dic[line_PID]
            list1D_RAA = RAA_test_dic[line_PID]
            list1D_RSA = RSA_test_dic[line_PID]
            list1D_Pro2Vec_1D = Pro2Vec_1D_test_dic[line_PID]
            list1D_Anchor = Anchor_test_dic[line_PID]
            list1D_HSP = HSP_test_dic[line_PID]
            list1D_HYD = HYD_test_dic[line_PID]
            list1D_PKA = PKA_test_dic[line_PID]
            list1D_PHY_Char_1 = PHY_Char_test_dic_1[line_PID]
            list1D_PHY_Char_2 = PHY_Char_test_dic_2[line_PID]
            list1D_PHY_Char_3 = PHY_Char_test_dic_3[line_PID]
            list1D_PHY_Prop_1 = PHY_Prop_test_dic_1[line_PID]
            list1D_PHY_Prop_2 = PHY_Prop_test_dic_2[line_PID]
            list1D_PHY_Prop_3 = PHY_Prop_test_dic_3[line_PID]
            list1D_PHY_Prop_4 = PHY_Prop_test_dic_4[line_PID]
            list1D_PHY_Prop_5 = PHY_Prop_test_dic_5[line_PID]
            list1D_PHY_Prop_6 = PHY_Prop_test_dic_6[line_PID]
            list1D_PHY_Prop_7 = PHY_Prop_test_dic_7[line_PID]

            list1D_PSSM_1 = PSSM_test_dic_1[line_PID]
            list1D_PSSM_2 = PSSM_test_dic_2[line_PID]
            list1D_PSSM_3 = PSSM_test_dic_3[line_PID]
            list1D_PSSM_4 = PSSM_test_dic_4[line_PID]
            list1D_PSSM_5 = PSSM_test_dic_5[line_PID]
            list1D_PSSM_6 = PSSM_test_dic_6[line_PID]
            list1D_PSSM_7 = PSSM_test_dic_7[line_PID]
            list1D_PSSM_8 = PSSM_test_dic_8[line_PID]
            list1D_PSSM_9 = PSSM_test_dic_9[line_PID]
            list1D_PSSM_10 = PSSM_test_dic_10[line_PID]
            list1D_PSSM_11 = PSSM_test_dic_11[line_PID]
            list1D_PSSM_12 = PSSM_test_dic_12[line_PID]
            list1D_PSSM_13 = PSSM_test_dic_13[line_PID]
            list1D_PSSM_14 = PSSM_test_dic_14[line_PID]
            list1D_PSSM_15 = PSSM_test_dic_15[line_PID]
            list1D_PSSM_16 = PSSM_test_dic_16[line_PID]
            list1D_PSSM_17 = PSSM_test_dic_17[line_PID]
            list1D_PSSM_18 = PSSM_test_dic_18[line_PID]
            list1D_PSSM_19 = PSSM_test_dic_19[line_PID]
            list1D_PSSM_20 = PSSM_test_dic_20[line_PID]

            np3D_pro2vec = Pro2Vec_test_dic[line_PID]
        ECO_2DList.append(list1D_ECO)
        RAA_2DList.append(list1D_RAA)
        RSA_2DList.append(list1D_RSA)
        Pro2Vec_1D_2DList.append(list1D_Pro2Vec_1D)
        Anchor_2DList.append(list1D_Anchor)
        HSP_2DList.append(list1D_HSP)
        HYD_2DList.append(list1D_HYD)
        PKA_2DList.append(list1D_PKA)
        PHY_Char_2DList_1.append(list1D_PHY_Char_1)
        PHY_Char_2DList_2.append(list1D_PHY_Char_2)
        PHY_Char_2DList_3.append(list1D_PHY_Char_3)
        PHY_Prop_2DList_1.append(list1D_PHY_Prop_1)
        PHY_Prop_2DList_2.append(list1D_PHY_Prop_2)
        PHY_Prop_2DList_3.append(list1D_PHY_Prop_3)
        PHY_Prop_2DList_4.append(list1D_PHY_Prop_4)
        PHY_Prop_2DList_5.append(list1D_PHY_Prop_5)
        PHY_Prop_2DList_6.append(list1D_PHY_Prop_6)
        PHY_Prop_2DList_7.append(list1D_PHY_Prop_7)

        PSSM_2DList_1.append(list1D_PSSM_1)
        PSSM_2DList_2.append(list1D_PSSM_2)
        PSSM_2DList_3.append(list1D_PSSM_3)
        PSSM_2DList_4.append(list1D_PSSM_4)
        PSSM_2DList_5.append(list1D_PSSM_5)
        PSSM_2DList_6.append(list1D_PSSM_6)
        PSSM_2DList_7.append(list1D_PSSM_7)
        PSSM_2DList_8.append(list1D_PSSM_8)
        PSSM_2DList_9.append(list1D_PSSM_9)
        PSSM_2DList_10.append(list1D_PSSM_10)
        PSSM_2DList_11.append(list1D_PSSM_11)
        PSSM_2DList_12.append(list1D_PSSM_12)
        PSSM_2DList_13.append(list1D_PSSM_13)
        PSSM_2DList_14.append(list1D_PSSM_14)
        PSSM_2DList_15.append(list1D_PSSM_15)
        PSSM_2DList_16.append(list1D_PSSM_16)
        PSSM_2DList_17.append(list1D_PSSM_17)
        PSSM_2DList_18.append(list1D_PSSM_18)
        PSSM_2DList_19.append(list1D_PSSM_19)
        PSSM_2DList_20.append(list1D_PSSM_20)

        # print("np3D_pro2vec.shape: ",np3D_pro2vec.shape)
        # print("SplitPro2Vec(np3D_pro2vec, args): ",SplitPro2Vec(np3D_pro2vec, args))
        if((int)(args.num_feature) == 0 or (int)(args.num_feature) == 5):
            Pro2Vec_List.extend(SplitPro2Vec(np3D_pro2vec, args))
        label_1DList.extend(list1D_label_this_line)
    fin.close()

    ECO_3D_np = Convert2DListTo3DNp(args, ECO_2DList)
    RAA_3D_np = Convert2DListTo3DNp(args, RAA_2DList)
    RSA_3D_np = Convert2DListTo3DNp(args, RSA_2DList)
    Pro2Vec_1D_3D_np = Convert2DListTo3DNp(args, Pro2Vec_1D_2DList)
    Anchor_3D_np = Convert2DListTo3DNp(args, Anchor_2DList)
    HSP_3D_np = Convert2DListTo3DNp(args, HSP_2DList)
    HYD_3D_np = Convert2DListTo3DNp(args, HYD_2DList)
    PKA_3D_np = Convert2DListTo3DNp(args, PKA_2DList)
    PHY_Char_3D_np_1 = Convert2DListTo3DNp(args, PHY_Char_2DList_1)
    PHY_Char_3D_np_2 = Convert2DListTo3DNp(args, PHY_Char_2DList_2)
    PHY_Char_3D_np_3 = Convert2DListTo3DNp(args, PHY_Char_2DList_3)
    PHY_Prop_3D_np_1 = Convert2DListTo3DNp(args, PHY_Prop_2DList_1)
    PHY_Prop_3D_np_2 = Convert2DListTo3DNp(args, PHY_Prop_2DList_2)
    PHY_Prop_3D_np_3 = Convert2DListTo3DNp(args, PHY_Prop_2DList_3)
    PHY_Prop_3D_np_4 = Convert2DListTo3DNp(args, PHY_Prop_2DList_4)
    PHY_Prop_3D_np_5 = Convert2DListTo3DNp(args, PHY_Prop_2DList_5)
    PHY_Prop_3D_np_6 = Convert2DListTo3DNp(args, PHY_Prop_2DList_6)
    PHY_Prop_3D_np_7 = Convert2DListTo3DNp(args, PHY_Prop_2DList_7)

    PSSM_3D_np_1 = Convert2DListTo3DNp(args, PSSM_2DList_1)
    PSSM_3D_np_2 = Convert2DListTo3DNp(args, PSSM_2DList_2)
    PSSM_3D_np_3 = Convert2DListTo3DNp(args, PSSM_2DList_3)
    PSSM_3D_np_4 = Convert2DListTo3DNp(args, PSSM_2DList_4)
    PSSM_3D_np_5 = Convert2DListTo3DNp(args, PSSM_2DList_5)
    PSSM_3D_np_6 = Convert2DListTo3DNp(args, PSSM_2DList_6)
    PSSM_3D_np_7 = Convert2DListTo3DNp(args, PSSM_2DList_7)
    PSSM_3D_np_8 = Convert2DListTo3DNp(args, PSSM_2DList_8)
    PSSM_3D_np_9 = Convert2DListTo3DNp(args, PSSM_2DList_9)
    PSSM_3D_np_10 = Convert2DListTo3DNp(args, PSSM_2DList_10)
    PSSM_3D_np_11 = Convert2DListTo3DNp(args, PSSM_2DList_11)
    PSSM_3D_np_12 = Convert2DListTo3DNp(args, PSSM_2DList_12)
    PSSM_3D_np_13 = Convert2DListTo3DNp(args, PSSM_2DList_13)
    PSSM_3D_np_14 = Convert2DListTo3DNp(args, PSSM_2DList_14)
    PSSM_3D_np_15 = Convert2DListTo3DNp(args, PSSM_2DList_15)
    PSSM_3D_np_16 = Convert2DListTo3DNp(args, PSSM_2DList_16)
    PSSM_3D_np_17 = Convert2DListTo3DNp(args, PSSM_2DList_17)
    PSSM_3D_np_18 = Convert2DListTo3DNp(args, PSSM_2DList_18)
    PSSM_3D_np_19 = Convert2DListTo3DNp(args, PSSM_2DList_19)
    PSSM_3D_np_20 = Convert2DListTo3DNp(args, PSSM_2DList_20)

    Pro2Vec_3D_np = np.asarray(Pro2Vec_List)
    print("Pro2Vec_3D_np.shape: ",Pro2Vec_3D_np.shape)
    label_2D_np = np.asarray(label_1DList).reshape(-1, 1)
    # print(label_2D_np)
    print("PKA_3D_np.shape: ", PKA_3D_np.shape)
    print("PHY_Char_3D_np_1.shape: ", PHY_Char_3D_np_1.shape)
    print("PHY_Prop_3D_np_7.shape: ", PHY_Prop_3D_np_7.shape)
    assert (ECO_3D_np.shape == RAA_3D_np.shape == RSA_3D_np.shape == Anchor_3D_np.shape == HYD_3D_np.shape == PKA_3D_np.shape == PHY_Char_3D_np_1.shape == PHY_Char_3D_np_2.shape == PHY_Char_3D_np_3.shape == PHY_Prop_3D_np_1.shape == PHY_Prop_3D_np_2.shape == PHY_Prop_3D_np_3.shape == PHY_Prop_3D_np_4.shape == PHY_Prop_3D_np_5.shape == PHY_Prop_3D_np_6.shape  == PHY_Prop_3D_np_7.shape == Pro2Vec_1D_3D_np.shape == HSP_3D_np.shape == PSSM_3D_np_20.shape == PSSM_3D_np_19.shape == PSSM_3D_np_1.shape)

    log_time("Preparing features")
    if ((int)(args.num_feature) == 0):
        logging.info("Using all features and pro2vec")
        all_features_3D_np = np.concatenate(
            (Pro2Vec_3D_np, ECO_3D_np, RAA_3D_np, RSA_3D_np, Pro2Vec_1D_3D_np, Anchor_3D_np, HSP_3D_np, HYD_3D_np, PKA_3D_np, PHY_Char_3D_np_1, PHY_Char_3D_np_2, PHY_Char_3D_np_3, PHY_Prop_3D_np_1, PHY_Prop_3D_np_2, PHY_Prop_3D_np_3, PHY_Prop_3D_np_4, PHY_Prop_3D_np_5, PHY_Prop_3D_np_6, PHY_Prop_3D_np_7), axis=2)
    elif ((int)(args.num_feature) == -1):
        logging.info("Using all features except pro2vec")
        all_features_3D_np = np.concatenate(
            (ECO_3D_np, RAA_3D_np, RSA_3D_np, Pro2Vec_1D_3D_np, Anchor_3D_np, HSP_3D_np, HYD_3D_np, PKA_3D_np, PHY_Char_3D_np_1, PHY_Char_3D_np_2, PHY_Char_3D_np_3, PHY_Prop_3D_np_1, PHY_Prop_3D_np_2, PHY_Prop_3D_np_3, PHY_Prop_3D_np_4, PHY_Prop_3D_np_5, PHY_Prop_3D_np_6, PHY_Prop_3D_np_7, PSSM_3D_np_1, PSSM_3D_np_2, PSSM_3D_np_3, PSSM_3D_np_4, PSSM_3D_np_5, PSSM_3D_np_6, PSSM_3D_np_7, PSSM_3D_np_8, PSSM_3D_np_9, PSSM_3D_np_10, PSSM_3D_np_11, PSSM_3D_np_12, PSSM_3D_np_13, PSSM_3D_np_14, PSSM_3D_np_15, PSSM_3D_np_16, PSSM_3D_np_17, PSSM_3D_np_18, PSSM_3D_np_19, PSSM_3D_np_20, ), axis=2)
    elif ((int)(args.num_feature) == 1):
        logging.info("Using only ECO")
        all_features_3D_np = ECO_3D_np
    elif ((int)(args.num_feature) == 2):
        logging.info("Using only RAA")
        all_features_3D_np = RAA_3D_np
    elif ((int)(args.num_feature) == 3):
        logging.info("Using only RSA")
        all_features_3D_np = RSA_3D_np
    elif ((int)(args.num_feature) == 4):
        logging.info("Using only Anchor")
        all_features_3D_np = Anchor_3D_np
    elif ((int)(args.num_feature) == 5):
        logging.info("Using only Pro2Vec")
        all_features_3D_np = Pro2Vec_3D_np
    elif ((int)(args.num_feature) == 6):
        logging.info("Using only HYD")
        all_features_3D_np = HYD_3D_np
    elif ((int)(args.num_feature) == 7):
        logging.info("Using only PKA")
        all_features_3D_np = PKA_3D_np
    elif ((int)(args.num_feature) == 8):
        logging.info("Using only HSP")
        all_features_3D_np = HSP_3D_np
    else:
        logging.error("option nf --num_feature is invalid")

    return all_features_3D_np, label_2D_np


def main():
    logging.basicConfig(format='[%(levelname)s] line %(lineno)d: %(message)s', level='INFO')
    print("testing:xxx",flush=True)
    log_time("Program started")
    time_start = time.time()
    parser = GetProgramArguments()
    args = parser.parse_args()
    # create directory
    MakeDir('plots/', args)
    MakeDir('logs/', args)
    MakeDir('savedModel/', args)
    # proteins_in_testing = CheckTrainAndTestDataSet(args)
    print("program arguments are: ", args)
    LoadFeatures(args)
    if (args.load_data == 0):
        log_time("Constructing training data")
        train_all_features_np3D, train_label_np_2D = LoadLabelsAndFormatFeatures(args, args.training_protein_fn, True)
        if (args.save_model == 1):
            np.save("savedModel/saved_train_validation_data/train_validation_feature_np_3D.npy", train_all_features_np3D)
            np.save("savedModel/saved_train_validation_data/train_validation_label_np_2D.npy", train_label_np_2D)
            log_time("loading saved_data done")
    else: # load data directly
        log_time("Loading training data directly")
        train_all_features_np3D = np.load("savedModel/saved_train_validation_data/train_validation_feature_np_3D.npy")
        train_label_np_2D = np.load("savedModel/saved_train_validation_data/train_validation_label_np_2D.npy")
    test_all_features_np3D, test_label_np_2D = LoadLabelsAndFormatFeatures(args, args.testing_protein_fn, False)

    # exit(1)
    # print("Dict_3mer_to_100vec size: ", len(Dict_3mer_to_100vec))
    TrainModel(args, train_all_features_np3D, train_label_np_2D, test_all_features_np3D, test_label_np_2D)
    log_time("Program ended")


def Build_ND_DB(feature_name, dimension):
    dic = {
        "DS_72": CUR_DIR + "../Features/" + feature_name + "/DS_72" + "_" + str(dimension) + ".txt",
        "DS_164": CUR_DIR + "../Features/" + feature_name + "/DS_164" + "_" + str(dimension) + ".txt",
        "DS_186": CUR_DIR + "../Features/" + feature_name + "/DS_186" + "_" + str(dimension) + ".txt",
        "SCRIBER_test": CUR_DIR + "../Features/" + feature_name + "/SCRIBER_test" + "_" + str(dimension) + ".txt",
        "SCRIBER_train": CUR_DIR + "../Features/" + feature_name + "/SCRIBER_train" + "_" + str(dimension) + ".txt",
        "survey_test": CUR_DIR + "../Features/" + feature_name + "/survey_test" + "_" + str(dimension) + ".txt",
        "survey_train": CUR_DIR + "../Features/" + feature_name + "/survey_train" + "_" + str(dimension) + ".txt",
    }
    return dic


def Build_1D_DB(feature_name):
    dic = {
        "DS_72": CUR_DIR + "../Features/" + feature_name + "/DS_72.txt",
        "DS_164": CUR_DIR + "../Features/" + feature_name + "/DS_164.txt",
        "DS_186": CUR_DIR + "../Features/" + feature_name + "/DS_186.txt",
        "SCRIBER_test": CUR_DIR + "../Features/" + feature_name + "/SCRIBER_test.txt",
        "SCRIBER_train": CUR_DIR + "../Features/" + feature_name + "/SCRIBER_train.txt",
        "survey_test": CUR_DIR + "../Features/" + feature_name + "/survey_test.txt",
        "survey_train": CUR_DIR + "../Features/" + feature_name + "/survey_train.txt",
    }
    return dic

# global variables
CUR_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
# a dictionary that stores <3mer, np of shape (1,1,100)>
Dict_3mer_to_100vec = {}
HYD_DB = Build_1D_DB("HYD")
PKA_DB = Build_1D_DB("PKA")
ECO_DB = Build_1D_DB("ECO")
RAA_DB = Build_1D_DB("RAA")
RSA_DB = Build_1D_DB("RSA")
Anchor_DB = Build_1D_DB("Anchor")
Pro2Vec_1D_DB = Build_1D_DB("Pro2Vec_1D")
HSP_DB = Build_1D_DB("HSP")

PHY_Char_DB_1 = Build_ND_DB("PHY_Char", 1)
PHY_Char_DB_2 = Build_ND_DB("PHY_Char", 2)
PHY_Char_DB_3 = Build_ND_DB("PHY_Char", 3)
PHY_Prop_DB_1 = Build_ND_DB("PHY_Prop", 1)
PHY_Prop_DB_2 = Build_ND_DB("PHY_Prop", 2)
PHY_Prop_DB_3 = Build_ND_DB("PHY_Prop", 3)
PHY_Prop_DB_4 = Build_ND_DB("PHY_Prop", 4)
PHY_Prop_DB_5 = Build_ND_DB("PHY_Prop", 5)
PHY_Prop_DB_6 = Build_ND_DB("PHY_Prop", 6)
PHY_Prop_DB_7 = Build_ND_DB("PHY_Prop", 7)

PSSM_DB_1 = Build_ND_DB("PSSM", 1)
PSSM_DB_2 = Build_ND_DB("PSSM", 2)
PSSM_DB_3 = Build_ND_DB("PSSM", 3)
PSSM_DB_4 = Build_ND_DB("PSSM", 4)
PSSM_DB_5 = Build_ND_DB("PSSM", 5)
PSSM_DB_6 = Build_ND_DB("PSSM", 6)
PSSM_DB_7 = Build_ND_DB("PSSM", 7)
PSSM_DB_8 = Build_ND_DB("PSSM", 8)
PSSM_DB_9 = Build_ND_DB("PSSM", 9)
PSSM_DB_10 = Build_ND_DB("PSSM", 10)
PSSM_DB_11 = Build_ND_DB("PSSM", 11)
PSSM_DB_12 = Build_ND_DB("PSSM", 12)
PSSM_DB_13 = Build_ND_DB("PSSM", 13)
PSSM_DB_14 = Build_ND_DB("PSSM", 14)
PSSM_DB_15 = Build_ND_DB("PSSM", 15)
PSSM_DB_16 = Build_ND_DB("PSSM", 16)
PSSM_DB_17 = Build_ND_DB("PSSM", 17)
PSSM_DB_18 = Build_ND_DB("PSSM", 18)
PSSM_DB_19 = Build_ND_DB("PSSM", 19)
PSSM_DB_20 = Build_ND_DB("PSSM", 20)

ECO_train_dic = {}
ECO_test_dic = {}
RAA_train_dic = {}
RAA_test_dic = {}
RSA_train_dic = {}
RSA_test_dic = {}
Pro2Vec_1D_train_dic = {}
Pro2Vec_1D_test_dic = {}
Anchor_train_dic = {}
Anchor_test_dic = {}
HSP_train_dic = {}
HSP_test_dic = {}
Pro2Vec_train_dic = {}
Pro2Vec_test_dic = {}
HYD_train_dic = {}
HYD_test_dic = {}
PKA_train_dic = {}
PKA_test_dic = {}
PHY_Char_train_dic_1 = {}
PHY_Char_train_dic_2 = {}
PHY_Char_train_dic_3 = {}
PHY_Prop_train_dic_1 = {}
PHY_Prop_train_dic_2 = {}
PHY_Prop_train_dic_3 = {}
PHY_Prop_train_dic_4 = {}
PHY_Prop_train_dic_5 = {}
PHY_Prop_train_dic_6 = {}
PHY_Prop_train_dic_7 = {}
PHY_Char_test_dic_1 = {}
PHY_Char_test_dic_2 = {}
PHY_Char_test_dic_3 = {}
PHY_Prop_test_dic_1 = {}
PHY_Prop_test_dic_2 = {}
PHY_Prop_test_dic_3 = {}
PHY_Prop_test_dic_4 = {}
PHY_Prop_test_dic_5 = {}
PHY_Prop_test_dic_6 = {}
PHY_Prop_test_dic_7 = {}

PSSM_train_dic_1 = {}
PSSM_train_dic_2 = {}
PSSM_train_dic_3 = {}
PSSM_train_dic_4 = {}
PSSM_train_dic_5 = {}
PSSM_train_dic_6 = {}
PSSM_train_dic_7 = {}
PSSM_train_dic_8 = {}
PSSM_train_dic_9 = {}
PSSM_train_dic_10 = {}
PSSM_train_dic_11 = {}
PSSM_train_dic_12 = {}
PSSM_train_dic_13 = {}
PSSM_train_dic_14 = {}
PSSM_train_dic_15 = {}
PSSM_train_dic_16 = {}
PSSM_train_dic_17 = {}
PSSM_train_dic_18 = {}
PSSM_train_dic_19 = {}
PSSM_train_dic_20 = {}
PSSM_test_dic_1 = {}
PSSM_test_dic_2 = {}
PSSM_test_dic_3 = {}
PSSM_test_dic_4 = {}
PSSM_test_dic_5 = {}
PSSM_test_dic_6 = {}
PSSM_test_dic_7 = {}
PSSM_test_dic_8 = {}
PSSM_test_dic_9 = {}
PSSM_test_dic_10 = {}
PSSM_test_dic_11 = {}
PSSM_test_dic_12 = {}
PSSM_test_dic_13 = {}
PSSM_test_dic_14 = {}
PSSM_test_dic_15 = {}
PSSM_test_dic_16 = {}
PSSM_test_dic_17 = {}
PSSM_test_dic_18 = {}
PSSM_test_dic_19 = {}
PSSM_test_dic_20 = {}

time_start = time.time()
time_end = time.time()

if __name__ == '__main__':
    main()
