import json
import re
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
            out_fn = out_base_fn + str(i) + ".txt"
            fout = open(out_fn, "a+")
            fout.write(">" + Pid + "\n")
            fout.write(line_Pseq + "\n")
            fout.write(",".join(map(str,pssm_np_2D[i-1])) + "\n")
            fout.close()
    else:
        print("length doesn't match for protein ", Pid)
        print("PSSM file has ", seq_len," lines, but sequence length is ", len(line_Pseq))

def get_sequence(pssm_fn):
    seq = ""
    fin = open(pssm_fn, "r")
    lines = fin.readlines()
    for line in lines:
        if re.match(r"^\ *\d+.*$",line):
            AA = line.split()[1]
            # print("AA: ",AA)
            seq = seq + AA
    fin.close
    return seq.rstrip("\n")

def load_DB(pssm_db_dir, output_dic_fn):
    print("loading PSSM DB..")
    dic_seq2_pssm_path = {}
    for file in os.listdir(pssm_db_dir):
        if file.endswith(".pssm"):
            # print ("file: ", file)
            abspath=os.path.abspath(file)
            # print(abspath)
            seq = get_sequence(abspath)
            # print (seq)
            dic_seq2_pssm_path[seq] = abspath
    print("Writing DB to json file")
    with open('/work2/DELPHI_Server/PSSM_database/PSSM_dic_seq2_pssm_path.json', 'w') as fp:
        json.dump(dic_seq2_pssm_path, fp)

def main():
    input_fn = sys.argv[1]
    PSSM_raw_dir= sys.argv[2] # move PSSMs there
    with open('/work2/DELPHI_Server/PSSM_database/PSSM_dic_seq2_pssm_path.json', 'r') as fp:
        dic_seq2_pssm_path = json.load(fp)
    fin = open(input_fn, "r")
    lines = fin.readlines()
    pid=""
    for line in lines:
        if line[0] == '>':
            pid = line.rstrip("\n")[1:]
        elif line[0] != '>':
            p_seq = line.rstrip("\n")
            if p_seq in dic_seq2_pssm_path:
                print(pid," is in DELPHI's PSSM database, loading it")
                cmd = "mkdir -p " + PSSM_raw_dir
                os.system(cmd)
                cmd = "cp " + dic_seq2_pssm_path[p_seq] + " " + PSSM_raw_dir + "/" + pid + ".fasta.pssm"

                os.system(cmd)
            else:
                print(pid," is not in DELPHI's PSSM database, will compute it")

        
    # pssm_db_dir = sys.argv[1] # dir where PSSMs are
    # output_dic_fn= sys.argv[2] # output fn of a python dictionary {seq, pssm_path}
    # load_DB(pssm_db_dir, output_dic_fn)
    # print("max_value: ", max_value)
    # print("min_value: ", min_value)

if __name__ == '__main__':
    main()

