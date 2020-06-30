import json
import re
import os
import numpy as np
import time
import sys
import math

def check(seq_fn):
    fin = open(seq_fn, "r")
    # fout = open(out_fn, "w")
    num_ori_seq = 0
    num_output_seq = 0
    while True:
        Pid = fin.readline().rstrip("\n")
        line_Pseq = fin.readline().rstrip("\n")
        if not line_Pseq:
            break
        if (Pid[0] != ">"):
            print("[Input sequence error:] Each sequence must have two lines: 1. >Protein_id 2. Protein_sequence")
            exit(1)
        if (" " in Pid or "\t" in Pid or "|" in Pid):
            print("[Error:] Unsupported characters in protein id: space, tab or |", Pid[1:])
            exit(1)
        num_ori_seq += 1
        if(len(line_Pseq) < 31):
            print("[Input sequence error:] The input sequences are required to have length > 31. Failed protein id: ", Pid[1:])
            exit(1)
            num_output_seq += 1
    fin.close()

def main():
    input_fn = sys.argv[1]
    check(input_fn)
        
if __name__ == '__main__':
    main()

