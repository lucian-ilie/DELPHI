from Bio.SubsMat import MatrixInfo as matlist
import os
import numpy as np
from pandas import DataFrame
import time
import sys
import math

PAM120 = matlist.pam120
train_pid_2_seq = {}
train_pid_2_label = {}
target_pid_2_seq = {}
target_pid_2_score = {}

def ReadTrain(train_fn):
    # pid, seq, label
    fin = open(train_fn, "r")
    while True:
        line_Pid = fin.readline().rstrip('\n').rstrip(' ')[1:]
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        line_label = fin.readline().rstrip('\n').rstrip(' ')
        if not line_label:
            break
        train_pid_2_seq[line_Pid] = line_Pseq
        train_pid_2_label[line_Pid] = line_label
    fin.close()

def ReadTarget(target_fn):
    # pid, seq
    fin = open(target_fn, "r")
    while True:
        line_Pid = fin.readline().rstrip('\n').rstrip(' ')[1:]
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break
        target_pid_2_seq[line_Pid] = line_Pseq
        target_pid_2_score[line_Pid] = np.zeros((len(line_Pseq)))
    fin.close()


def ReadHSP(HSP_fn, target_fn, out_fn):
    # > p1 and p2
    # sta1 sta2 length
    max_value = 468
    fin = open(HSP_fn, "r")
    p_train = ""
    p_target = ""
    need_swap = 0
    need_read = 0
    lines = fin.readlines()
    for line in lines:
        line=line.rstrip('\n').rstrip(' ')
        if (line[0] == ">"):  # read ids
            # print("line.split")
            # print(line.split(' '))
            p1 = line.split(' ')[1]
            p2 = line.split(' ')[3]
            if (p1 in train_pid_2_seq and p2 in target_pid_2_seq):
                p_train = p1
                p_target = p2
                need_read = 1
                need_swap = 0
            elif (p2 in train_pid_2_seq and p1 in target_pid_2_seq):
                p_train = p2
                p_target = p1
                need_read = 1
                need_swap = 1
            else:
                need_read = 0
                # print("no need to read")
                # print(p1)
                # print(p2)
        else:  # read HSPs
            if (need_read):
                sta_train = (int)(line.split(' ')[0])
                sta_target = (int)(line.split(' ')[1])
                hsp_length = (int)(line.split(' ')[2])
                # print("sta_train: ",sta_train)
                # print("sta_target: ",sta_target)
                # print("hsp_length: ",hsp_length)
                if (need_swap):
                    temp = sta_train
                    sta_train = sta_target
                    sta_target = temp
                for i in range(hsp_length):
                    position_train = sta_train + i
                    position_target = sta_target + i
                    if ((position_train < len(train_pid_2_label[p_train])) and (position_target < len(target_pid_2_seq[p_target])) and (train_pid_2_seq[p_train][position_train] != 'U') and (target_pid_2_seq[p_target][position_target] != 'U')):
                        if (train_pid_2_label[p_train][position_train] == "1"):
                            # print("is 1")
                            # print("score: ", PAM120[(
                            #         train_pid_2_seq[p_train][position_train],
                            #         target_pid_2_seq[p_target][position_target])])
                            # add score if similar, don't deduct score
                            if((train_pid_2_seq[p_train][position_train],target_pid_2_seq[p_target][position_target]) in PAM120):
                                score = PAM120[(
                                    train_pid_2_seq[p_train][position_train],
                                    target_pid_2_seq[p_target][position_target])]
                            else:
                                score = PAM120[(
                                    target_pid_2_seq[p_target][position_target],
                                    train_pid_2_seq[p_train][position_train])]
                            target_pid_2_score[p_target][position_target] = max(score, 0) + target_pid_2_score[p_target][position_target]
                            # max_value = max(max_value, target_pid_2_score[p_target][position_target])
        # print("max value: ",max_value)
    fin.close()

    #normalize
    for key in target_pid_2_score:
        for i in range(target_pid_2_score[key].shape[0]):
            if (max_value == 0):
                target_pid_2_score[key][i] = 0
            else:
                target_pid_2_score[key][i] = (target_pid_2_score[key][i])/max_value

    #print
    # pid, seq
    fin = open(target_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline().rstrip('\n').rstrip(' ')[1:]
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break
        fout.write(">" + line_Pid + "\n")
        fout.write(line_Pseq + "\n")
        fout.write(",".join(map(str, target_pid_2_score[line_Pid])) + "\n")
    fin.close()
    fout.close()


def main():
    # print("start")
    train_fn = sys.argv[1] # train fasta with label
    target_fn = sys.argv[2] # target fasta file. Produce the HSP feature of this file
    HSP_fn = sys.argv[3] # HSP file name. the HSP file contains all HSPs involving train and target proteins. No self HSP is included
    out_fn = sys.argv[4]
    ReadTrain(train_fn)
    ReadTarget(target_fn)
    ReadHSP(HSP_fn, target_fn, out_fn)
    # print("end")

if __name__ == '__main__':
    main()

