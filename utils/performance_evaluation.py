# read prediction results using DLpred on SCIRBER dataset
#argv[1]: label file
#argv[2]: prediction result file
import math
import os
import sys
from os import path
import numpy as np
import csv
import matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve
matplotlib.use('pdf')


# dim: delimiter
def PlotRocAndPRCurvesAndMetrics(truth, pred, args_prefix, csvPre=""):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    # print("fpr: ", fpr)
    # print("tpr: ", tpr)
    # print("thresholds: ", thresholds)

    # ROC curve
    au_roc = auc(fpr, tpr)
    print("Area under ROC curve: ", au_roc)
    # plt.title(args.prefix + ' ROC curve')
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # # plt.show()
    # plt.savefig("plots/" + args.prefix + "/ROC_" + csvPre + ".pdf")
    # plt.close()

    # PR curve
    precision, recall, thresholds = precision_recall_curve(truth, pred)
    aupr = auc(recall, precision)
    print("Area under PR curve: ", aupr)
    # plt.title(args.prefix + ' PR curve')
    # plt.plot(recall, precision)
    # # plt.show()
    # plt.savefig("plots/" + args.prefix + "/PR_" + csvPre + ".pdf")
    # plt.close()

    # evaluation metrics
    # step 1: calculate the threshold then convert score to binary number
    sorted_pred = np.sort(pred)
    sorted_pred_descending = np.flip(sorted_pred)  # from big to small
    num_of_1 = np.count_nonzero(truth)
    threshold = sorted_pred_descending.item(num_of_1 - 1)
    print("threshold: ",threshold)
    pred_binary = np.where(pred >= threshold, 1, 0)
    TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1_score, accuracy = CalculateEvaluationMetrics(truth, pred_binary)
    PrintToCSV(csvPre + args_prefix, au_roc, aupr, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC,
                   threshold, F1_score, accuracy)
    return au_roc
# write to csv file csv/results.csv.
def PrintToCSV(prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold, F1_score, accuracy):
    # header = ['prefix', 'AUC', 'AUPR', 'TP', 'FP', 'TN', 'FN', 'sensitivity', 'specificity', 'recall', 'precision',
    #           'MCC', 'threshold', 'F1_score']
    row = [prefix, sensitivity, specificity, precision, accuracy, F1_score, MCC, AUC, AUPR]
    # row = [prefix, AUC, AUPR, TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, threshold, F1_score]
    with open('result.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        # writer.writerow(header)
        writer.writerow(row)
    csvFile.close()

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
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    # same as sensitivity
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("MCC: ", MCC)
    print("F1: ", F1)
    return TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1, accuracy

def get_array_of_float_from_a_line(line, dim):
    res = []
    line = line.rstrip('\n').rstrip(' ').rstrip(',').split(dim)
    # print("line: ",line)
    res += [float(i) for i in line]
    return res

def get_array_of_int_from_a_line(line):
    res = []
    for i in line.rstrip('\n').rstrip(' '):
        if (i == '.' or i == '0'):
            res.append(0)
        else:
            res.append(1)
    return res

dic_pid_2_score={}
dic_pid_2_label={}

def read_result_file_JianZhang_format(fn, if_append_8_zeros_head_and_tail):
    values = []
    if (if_append_8_zeros_head_and_tail):
        values = [0,0,0,0]


    fin = open(fn)
    lines = fin.readlines()
    for line in lines:
        if (line[0] != "#"):
            val = float(line.rstrip('\n').split()[2])
            values.append(val)
    if (if_append_8_zeros_head_and_tail):
        values.extend([0,0,0,0])
    return  values

def LoadTestingLabel(label_fn, result_DIR):
    #read label file to dic
    all_labels = []
    all_values = []
    fin_label = open(label_fn, "r")
    num_pro = 0
    while True:
        line_PID = fin_label.readline().rstrip('\n')[1:].upper()
        line_Seq = fin_label.readline().rstrip('\n')
        line_label = fin_label.readline().rstrip('\n')
        if not line_label:
            break
        cur_label = get_array_of_int_from_a_line(line_label)
        result_fn = result_DIR+"/"+line_PID+".txt"
        if path.exists(result_fn):
            if(count_num_of_lines_exclude_end_of_line_in_a_file(result_fn) == len(cur_label) - 8):

                cur_value = read_result_file_JianZhang_format(result_fn, True)
                # print("cur_value: ", cur_value)

                # cur_label = cur_label[4:]# remove first 4
                # cur_label = cur_label[0:len(cur_label)-4] #remove last 4
                # print(cur_label)
            elif (count_num_of_lines_exclude_end_of_line_in_a_file(result_fn) != len(cur_label)):
                print("label and value length not eaqual && label != value - 8: ", result_fn)
                exit(1)
            else:
                cur_value = read_result_file_JianZhang_format(result_fn,False)
            if((len(cur_value) != len(cur_label))):
                print(line_PID)
                print("len(cur_value): ",len(cur_value))
                print("len(cur_label): ",len(cur_label))
                exit()
            all_labels.extend(cur_label)
            all_values.extend(cur_value)
            # if (line_PID == "P80563"):
                # print("cur_label",cur_label)
                # print("cur_value",cur_value)
            num_pro += 1

        else:
            print (result_fn," doesn't exist")
            # exit(1)
    print("num_pro results readed: ",num_pro)


    fin_label.close()

    return all_labels, all_values


dic_non_similar_proteins = {}

def count_num_of_lines_exclude_end_of_line_in_a_file(fn):
    count = 0
    with open(fn, 'r') as f:
        for line in f:
            if (line[0] != "#"):
                if (line.rstrip('\n').rstrip('\r') != ''):
                    count += 1
    f.close()
    return  count

def main():

    # print(count_num_of_lines_exclude_end_of_line_in_a_file("/home/j00492398/test_joey/interface-pred/workspace/compute_evaluation_metrics/results_raw_format/DS186/CRF-PPI/1GL4A.txt"))
    # exit(0)

    result_DIR=sys.argv[1]
    label_fn=sys.argv[2]
    prefix=sys.argv[3]
    label, value = LoadTestingLabel(label_fn, result_DIR)
    label_np = np.array(label)
    value_np = np.array(value)
    # print ("label_np: ",label_np.ravel().shape)
    # print ("value_np: ",value_np.ravel().shape)
    # print ("label_np.ravel()[:10]: ",label_np.ravel()[:100])
    # print ("value_np.ravel()[:10]: ",value_np.ravel()[:100])
    PlotRocAndPRCurvesAndMetrics(truth=label_np.ravel(), pred=value_np.ravel(), args_prefix=prefix,csvPre="")


if __name__ == '__main__':
    main()
