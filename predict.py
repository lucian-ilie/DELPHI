import numpy as np
import os
import logging
import datetime
from keras.models import load_model
import time
import argparse

WINDOW_SIZE=31

def Predict(args, test_all_features_np3D):
    log_time("Start predicting")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if (args.cpu == 1):
        log_time("using the CPU model")
        model = load_model("models/DELPHI_cpu.h5")
    elif (args.cpu == 0):
        log_time("using the GPU model")
        model = load_model("models/DELPHI_gpu.h5")
    else:
        print("{Error]: invalid option! -c can be either 1:cpu or 0:gpu")
        exit(1)
    log_time("Start predicting...")
    y_pred_testing = model.predict(test_all_features_np3D, batch_size=1024).ravel()

    # load input proteins again and output the predict values
    start_index = 0
    fin = open(args.input_fn, "r")
    while True:
        line_PID = fin.readline()[1:].rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break
        if (len(line_Pseq) < 32):
            print("The input sequence is shorter than 32. ", line_PID)
            exit(1)
        fout = open(args.out_dir+"/"+line_PID.upper()+".txt", "w")
        fout.write("# Prediction results by DELPHI\n")
        fout.write("# Output columns:\n")
        fout.write("# Index: the position of the residue in the input sequence, starting from 1\n")
        fout.write("# Residue: the amino acid residue at the position 'index', represented using 1-letter code\n")
        fout.write("# DELPHI prediction value: the DELPHI prediction value for the residue at the position 'index'.\n")
        fout.write("# [Index]\t[Residue]\t[DELPHI prediction value]\n")
        for i in range(len(line_Pseq)):
            fout.write(str(i + 1) + "\t" + line_Pseq[i] + "\t" + str(y_pred_testing[start_index + i]) + "\n")
        fout.close()
        start_index += len(line_Pseq)
    fin.close()
    log_time("End predicting")


def GetProgramArguments():
    logging.info("Parsing program arguments...")
    parser = argparse.ArgumentParser(description='predict.py')
    parser._action_groups.pop()

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input_fn', type=str, required=True,
                          help='str: input protein sequences. In fasta format')
    required.add_argument('-o', '--out_dir', type=str, required=True,
                          help='str: output directory')
    required.add_argument('-d', '--tmp_dir', type=str, required=True,
                          help='str: temporary  directory to store all features. Will be deleted at the end of the program')

    # optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-c', '--cpu', type=int, default=1,
                          help='int: use cpu or gpu. 1: cpu; 0: gpu. Default: 1')
    return parser

def log_time(prefix):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    logging.info("%s: %s", prefix, st)

# dim: delimiter
def get_array_of_float_from_a_line(line, dim):
    res = []
    line = line.rstrip('\n').rstrip(' ').split(dim)
    # print (line)
    res += [float(i) for i in line]
    return res

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

# load each feature dictionary
def LoadFeatures(args):
    log_time("Loading feature ECO")
    Read1DFeature(args.tmp_dir+"/ECO.txt", ECO_test_dic)
    log_time("Loading feature RAA")
    Read1DFeature(args.tmp_dir+"/RAA.txt", RAA_test_dic)
    log_time("Loading feature RSA")
    Read1DFeature(args.tmp_dir+"/RSA.txt", RSA_test_dic)
    log_time("Loading feature Anchor")
    Read1DFeature(args.tmp_dir+"/Anchor.txt", Anchor_test_dic)
    log_time("Loading feature HYD")
    Read1DFeature(args.tmp_dir+"/HYD.txt", HYD_test_dic)
    log_time("Loading feature PKA")
    Read1DFeature(args.tmp_dir+"/PKA.txt", PKA_test_dic)
    log_time("Loading feature Pro2Vec_1D")
    Read1DFeature(args.tmp_dir+"/Pro2Vec_1D.txt", Pro2Vec_1D_test_dic)
    log_time("Loading feature HSP")
    Read1DFeature(args.tmp_dir+"/HSP.txt", HSP_test_dic)
    log_time("Loading feature POSITION")
    Read1DFeature(args.tmp_dir+"/POSITION.txt", POSITION_test_dic)

    log_time("Loading feature PHY_Char")
    Read1DFeature(args.tmp_dir+"/PHY_Char1.txt", PHY_Char_test_dic_1)
    Read1DFeature(args.tmp_dir+"/PHY_Char2.txt", PHY_Char_test_dic_2)
    Read1DFeature(args.tmp_dir+"/PHY_Char3.txt", PHY_Char_test_dic_3)

    log_time("Loading feature PHY_Prop")
    Read1DFeature(args.tmp_dir+"/PHY_Prop1.txt", PHY_Prop_test_dic_1)
    Read1DFeature(args.tmp_dir+"/PHY_Prop2.txt", PHY_Prop_test_dic_2)
    Read1DFeature(args.tmp_dir+"/PHY_Prop3.txt", PHY_Prop_test_dic_3)
    Read1DFeature(args.tmp_dir+"/PHY_Prop4.txt", PHY_Prop_test_dic_4)
    Read1DFeature(args.tmp_dir+"/PHY_Prop5.txt", PHY_Prop_test_dic_5)
    Read1DFeature(args.tmp_dir+"/PHY_Prop6.txt", PHY_Prop_test_dic_6)
    Read1DFeature(args.tmp_dir+"/PHY_Prop7.txt", PHY_Prop_test_dic_7)

    log_time("Loading feature PSSM")
    Read1DFeature(args.tmp_dir+"/PSSM1.txt", PSSM_test_dic_1)
    Read1DFeature(args.tmp_dir+"/PSSM2.txt", PSSM_test_dic_2)
    Read1DFeature(args.tmp_dir+"/PSSM3.txt", PSSM_test_dic_3)
    Read1DFeature(args.tmp_dir+"/PSSM4.txt", PSSM_test_dic_4)
    Read1DFeature(args.tmp_dir+"/PSSM5.txt", PSSM_test_dic_5)
    Read1DFeature(args.tmp_dir+"/PSSM6.txt", PSSM_test_dic_6)
    Read1DFeature(args.tmp_dir+"/PSSM7.txt", PSSM_test_dic_7)
    Read1DFeature(args.tmp_dir+"/PSSM8.txt", PSSM_test_dic_8)
    Read1DFeature(args.tmp_dir+"/PSSM9.txt", PSSM_test_dic_9)
    Read1DFeature(args.tmp_dir+"/PSSM10.txt", PSSM_test_dic_10)
    Read1DFeature(args.tmp_dir+"/PSSM11.txt", PSSM_test_dic_11)
    Read1DFeature(args.tmp_dir+"/PSSM12.txt", PSSM_test_dic_12)
    Read1DFeature(args.tmp_dir+"/PSSM13.txt", PSSM_test_dic_13)
    Read1DFeature(args.tmp_dir+"/PSSM14.txt", PSSM_test_dic_14)
    Read1DFeature(args.tmp_dir+"/PSSM15.txt", PSSM_test_dic_15)
    Read1DFeature(args.tmp_dir+"/PSSM16.txt", PSSM_test_dic_16)
    Read1DFeature(args.tmp_dir+"/PSSM17.txt", PSSM_test_dic_17)
    Read1DFeature(args.tmp_dir+"/PSSM18.txt", PSSM_test_dic_18)
    Read1DFeature(args.tmp_dir+"/PSSM19.txt", PSSM_test_dic_19)
    Read1DFeature(args.tmp_dir+"/PSSM20.txt", PSSM_test_dic_20)

    log_time("Loading features done")

# split a 1D list into window size. Shifting step is 1. For the beginning and end of the sequence, pad with 0
def Split1Dlist2NpArrays(args, inputList):
    list_1d_after_split = []
    win_size = WINDOW_SIZE
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
    input2DList_after_split = []
    for input1DList in input2DList:
        input1DList_after_split = Split1Dlist2NpArrays(args, input1DList)
        input2DList_after_split.extend(input1DList_after_split)

    np_2d_array = np.asarray(input2DList_after_split)
    np_2d_array = np_2d_array.reshape(-1, WINDOW_SIZE)
    return np_2d_array

def Convert2DListTo3DNp(args, list2D):
    np_2d = Split2DList2NpArrays(args, list2D)
    np_3D = np_2d.reshape(np_2d.shape[0], np_2d.shape[1], 1)
    return np_3D

def LoadLabelsAndFormatFeatures(args):
    ECO_2DList = []
    RAA_2DList = []
    RSA_2DList = []
    Pro2Vec_1D_2DList = []
    Anchor_2DList = []
    HSP_2DList = []
    HYD_2DList = []
    PKA_2DList = []
    POSITION_2DList = []
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

    fin = open(args.input_fn, "r")
    while True:
        line_PID = fin.readline().rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break

        list1D_ECO = ECO_test_dic[line_PID]
        list1D_RAA = RAA_test_dic[line_PID]
        list1D_RSA = RSA_test_dic[line_PID]
        list1D_Pro2Vec_1D = Pro2Vec_1D_test_dic[line_PID]
        list1D_Anchor = Anchor_test_dic[line_PID]
        list1D_HSP = HSP_test_dic[line_PID]
        list1D_HYD = HYD_test_dic[line_PID]
        list1D_PKA = PKA_test_dic[line_PID]
        list1D_POSITION = POSITION_test_dic[line_PID]
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

        ECO_2DList.append(list1D_ECO)
        RAA_2DList.append(list1D_RAA)
        RSA_2DList.append(list1D_RSA)
        Pro2Vec_1D_2DList.append(list1D_Pro2Vec_1D)
        Anchor_2DList.append(list1D_Anchor)
        HSP_2DList.append(list1D_HSP)
        HYD_2DList.append(list1D_HYD)
        PKA_2DList.append(list1D_PKA)
        POSITION_2DList.append(list1D_POSITION)
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

    fin.close()

    ECO_3D_np = Convert2DListTo3DNp(args, ECO_2DList)
    RAA_3D_np = Convert2DListTo3DNp(args, RAA_2DList)
    RSA_3D_np = Convert2DListTo3DNp(args, RSA_2DList)
    Pro2Vec_1D_3D_np = Convert2DListTo3DNp(args, Pro2Vec_1D_2DList)
    Anchor_3D_np = Convert2DListTo3DNp(args, Anchor_2DList)
    HSP_3D_np = Convert2DListTo3DNp(args, HSP_2DList)
    HYD_3D_np = Convert2DListTo3DNp(args, HYD_2DList)
    PKA_3D_np = Convert2DListTo3DNp(args, PKA_2DList)
    POSITION_3D_np = Convert2DListTo3DNp(args, POSITION_2DList)
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

    assert (ECO_3D_np.shape == RAA_3D_np.shape == RSA_3D_np.shape == Anchor_3D_np.shape == HYD_3D_np.shape == PKA_3D_np.shape == PHY_Char_3D_np_1.shape == PHY_Char_3D_np_2.shape == PHY_Char_3D_np_3.shape == PHY_Prop_3D_np_1.shape == PHY_Prop_3D_np_2.shape == PHY_Prop_3D_np_3.shape == PHY_Prop_3D_np_4.shape == PHY_Prop_3D_np_5.shape == PHY_Prop_3D_np_6.shape  == PHY_Prop_3D_np_7.shape == Pro2Vec_1D_3D_np.shape == HSP_3D_np.shape == PSSM_3D_np_20.shape == PSSM_3D_np_19.shape == PSSM_3D_np_1.shape == POSITION_3D_np.shape)

    log_time("Assembling 12 features")

    all_features_3D_np = np.concatenate(
            (ECO_3D_np, RAA_3D_np, RSA_3D_np, Pro2Vec_1D_3D_np, Anchor_3D_np, HSP_3D_np, HYD_3D_np, PKA_3D_np, PHY_Char_3D_np_1, PHY_Char_3D_np_2, PHY_Char_3D_np_3, PHY_Prop_3D_np_1, PHY_Prop_3D_np_2, PHY_Prop_3D_np_3, PHY_Prop_3D_np_4, PHY_Prop_3D_np_5, PHY_Prop_3D_np_6, PHY_Prop_3D_np_7, PSSM_3D_np_1, PSSM_3D_np_2, PSSM_3D_np_3, PSSM_3D_np_4, PSSM_3D_np_5, PSSM_3D_np_6, PSSM_3D_np_7, PSSM_3D_np_8, PSSM_3D_np_9, PSSM_3D_np_10, PSSM_3D_np_11, PSSM_3D_np_12, PSSM_3D_np_13, PSSM_3D_np_14, PSSM_3D_np_15, PSSM_3D_np_16, PSSM_3D_np_17, PSSM_3D_np_18, PSSM_3D_np_19, PSSM_3D_np_20, POSITION_3D_np), axis=2)

    return all_features_3D_np

def main():
    logging.basicConfig(format='[%(levelname)s] line %(lineno)d: %(message)s', level='INFO')
    log_time("Program started")
    parser = GetProgramArguments()
    args = parser.parse_args()
    print("program arguments are: ", args)
    LoadFeatures(args)
    test_all_features_np3D = LoadLabelsAndFormatFeatures(args)
    Predict(args, test_all_features_np3D)
    log_time("Program ended")

CUR_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

# a dictionary that stores <3mer, np of shape (1,1,100)>
Dict_3mer_to_100vec = {}
ECO_test_dic = {}
RAA_test_dic = {}
RSA_test_dic = {}
Pro2Vec_1D_test_dic = {}
Pro2Vec_embedding_test_dic = {}
Anchor_test_dic = {}
HSP_test_dic = {}
HYD_test_dic = {}
PKA_test_dic = {}
POSITION_test_dic = {}

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
