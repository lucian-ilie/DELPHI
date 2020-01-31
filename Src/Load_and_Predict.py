# fix the random see value so the results are re-producible
seed_value = 7
import numpy as np
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
import os
import csv
import logging
import datetime
from keras.models import Sequential, load_model
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

from Many2One_BiLSTM import CalculateEvaluationMetrics, get_array_of_float_from_a_line, get_array_of_int_from_a_line, Split1Dlist2NpArrays, Split2DList2NpArrays, PlotRocAndPRCurvesAndMetrics, Split2DNp3DNp, SplitPro2Vec2NpArrays, Convert2DListTo3DNp, PrintToCSV, MakeDir, PlotAccLossCurves, CountLabelIn2DList,log_time, get_3mer_and_np100vec_from_a_line, LoadProtVec3Grams, GetProVecFeature, CheckDiff, CheckTrainAndTestDataSet, Read1DFeature, PreparePro2Vec, ReadNDFeature, LoadFeatures, SplitPro2Vec, LoadLabelsAndFormatFeatures, Build_ND_DB, Build_1D_DB

def Predict(args, test_all_features_np3D, test_label_np_2D ):
    log_time("in TrainModel")

    cur_time = time.time()
    cur_time_formatted = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    tensorboard = TensorBoard(log_dir="tensorboard_log/{}".format(args.prefix))

    model = load_model(args.model_path)
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    # mc = ModelCheckpoint("savedModel/" + args.prefix + "/" + "best_model.h5", monitor='val_loss', mode='min',
    #                      verbose=1, save_best_only=True)

    log_time("Start predicting...")
    # if (os.path.isfile("savedModel/"+ args.prefix + "/train_feature.npy")):
    #     train_feature_np = np.load("/home/j00492398/test_joey/interface-pred/Src/savedModel/saved_train_validation_data/train_feature.npy")
    #     train_label_np = np.load("/home/j00492398/test_joey/interface-pred/Src/savedModel/saved_train_validation_data/train_label.npy")
    #     validation_feature_np = np.load("/home/j00492398/test_joey/interface-pred/Src/savedModel/saved_train_validation_data/validation_feature.npy")
    #     validation_label_np = np.load("/home/j00492398/test_joey/interface-pred/Src/savedModel/saved_train_validation_data/validation_label.npy")
    #     y_pred_testing = model.predict(train_feature_np, batch_size=args.batch_size).ravel()
    #     PlotRocAndPRCurvesAndMetrics(train_label_np.ravel(), y_pred_testing.ravel(), args, "train_ep"+str(args.epochs)+"_")
    #     y_pred_testing = model.predict(validation_feature_np, batch_size=args.batch_size).ravel()
    #     PlotRocAndPRCurvesAndMetrics(validation_label_np.ravel(), y_pred_testing.ravel(), args, "validation_ep"+str(args.epochs)+"_")

    y_pred_testing = model.predict(test_all_features_np3D, batch_size=args.batch_size).ravel()
    np.save("/home/j00492398/test_joey/interface-pred/workspace/plot_ROC_PR/DELPHI_label_"+args.testing_dataset_prefix+".npy", test_label_np_2D.ravel())
    np.save("/home/j00492398/test_joey/interface-pred/workspace/plot_ROC_PR/DELPHI_value_"+args.testing_dataset_prefix+".npy", y_pred_testing.ravel())

    PlotRocAndPRCurvesAndMetrics(test_label_np_2D.ravel(), y_pred_testing.ravel(), args, args.testing_dataset_prefix + "_ep"+str(args.epochs)+"_")
    del model


    # print("AUC[] for each cross validation: ", AUC)
    #
    # with open('csv/result.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     row = ["CV_AUC_mean_STD_" + args.prefix, np.mean(AUC), np.std(AUC)]
    #     writer.writerow(row)
    # csvFile.close()

def GetProgramArguments():
    logging.info("Parsing program arguments...")
    parser = argparse.ArgumentParser(description='BiLSTM_train_pred.py')
    parser._action_groups.pop()

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-mp', '--model_path', type=str, required=True,help='str: the path the a saved model"')
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
    # train_all_features_np3D, train_label_np_2D = LoadLabelsAndFormatFeatures(args, args.training_protein_fn, True)
    test_all_features_np3D, test_label_np_2D = LoadLabelsAndFormatFeatures(args, args.testing_protein_fn, False)

    # exit(1)
    # print("Dict_3mer_to_100vec size: ", len(Dict_3mer_to_100vec))
    Predict(args, test_all_features_np3D, test_label_np_2D)
    log_time("Program ended")



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

ECO_train_dic = {}
ECO_test_dic = {}
RAA_train_dic = {}
RAA_test_dic = {}
RSA_train_dic = {}
RSA_test_dic = {}
Pro2Vec_1D_train_dic = {}
Pro2Vec_1D_test_dic = {}
HSP_train_dic = {}
HSP_test_dic = {}
Anchor_train_dic = {}
Anchor_test_dic = {}
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
time_start = time.time()
time_end = time.time()

if __name__ == '__main__':
    main()
