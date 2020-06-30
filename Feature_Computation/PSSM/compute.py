import os
import numpy as np
import time
import sys
import math




def load_fasta_and_compute(seq_fn, out_base_fn, raw_pssm_dir):
    fin = open(seq_fn, "r")
    # fout = open(out_fn, "w")
    while True:
        Pid = fin.readline().rstrip("\n")[1:]
        line_Pseq = fin.readline().rstrip("\n")
        if not line_Pseq:
            break
        pssm_fn = raw_pssm_dir + "/" + Pid + ".fasta.pssm"
        LoadPSSMandPrintFeature(pssm_fn, out_base_fn, Pid, line_Pseq)
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



def LoadPSSMandPrintFeature(pssm_fn, out_base_fn, Pid, line_Pseq):
    print(Pid)
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
            out_fn = out_base_fn + str(i) + ".txt"
            fout = open(out_fn, "a+")
            fout.write(">" + Pid + "\n")
            fout.write(line_Pseq + "\n")
            fout.write(",".join(map(str,pssm_np_2D[i-1])) + "\n")
            fout.close()
    else:
        print("length doesn't match for protein ", Pid)
        print("PSSM file has ", seq_len," lines, but sequence length is ", len(line_Pseq))

def main():
    seq_fn = sys.argv[1]
    out_base_fn = sys.argv[2]
    raw_pssm_dir = sys.argv[3]
    load_fasta_and_compute(seq_fn, out_base_fn, raw_pssm_dir)
    print("max_value: ", max_value)
    print("min_value: ", min_value)

max_value = 13.0
min_value = -16.0
if __name__ == '__main__':
    main()

