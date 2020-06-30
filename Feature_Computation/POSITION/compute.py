import os
import numpy as np
import sys

def RetriveFeatureFromASequence(seq):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    Feature = []
    for index, item in enumerate(seq):
        Feature.append(float(index+1)/float(len(seq)))
    return Feature


def load_fasta_and_compute(seq_fn, out_fn):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        Feature = RetriveFeatureFromASequence(line_Pseq)
        fout.write(",".join(map(str,Feature)) + "\n")
    fin.close()
    fout.close()

def main():

    seq_fn = sys.argv[1]
    out_fn = sys.argv[2]
    load_fasta_and_compute(seq_fn, out_fn)



if __name__ == '__main__':
    main()

