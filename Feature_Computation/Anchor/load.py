import os
import numpy as np
import time
import sys
import argparse
import math

# dbDir is where RSA database. The DIR of where the RSA files should be loaded, remember to add '/'
def LoadRSA(seq_fn, dbDir, out_fn, max_rsa, min_rsa):
    max_value = 1
    min_value = 0
    fin_seq = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_PID = fin_seq.readline().rstrip("\n")[1:]
        line_Seq = fin_seq.readline().rstrip("\n")
        if not line_Seq:
            break

        rsa = []
        raw_feature_fn = dbDir +"/" + line_PID + ".fasta.anchor"
        try:
            fin = open(raw_feature_fn, "r")
        except Exception as e:
            print("open file failed. exit now: ", raw_feature_fn)
            exit(1)

        lines = fin.readlines()
        seq_len = len(line_Seq)
        for x in lines:
            if (not x.startswith("#")):
                value = float(x.split(' ')[2])
                # max_value = max(max_value, value)
                # min_value = min(min_value, value)
                value = (value - min_rsa) / (max_rsa - min_rsa)
                rsa.append(value)

        fin.close()
        # some proteins' rsa file fails to be generated, pad with 0
        if (len(rsa) == 0):
            rsa = [0] * seq_len
            print("[warning:]", line_PID, "has no feature file. Pad ", seq_len, " 0 for it.")
        fout.write(">" + line_PID + "\n")
        fout.write(line_Seq + "\n")
        fout.write(",".join(map(str, rsa)) + "\n")
    fin_seq.close()
    fout.close()



def main():
    seq_fn = sys.argv[1]
    raw_feature_dir = sys.argv[2]
    out_fn = sys.argv[3]

    LoadRSA(seq_fn, raw_feature_dir, out_fn, 1.0, 0.0)

if __name__ == '__main__':
    main()
