import numpy as np
import sys

def BuildRAADictionary():
    RAA_table = np.array(
        [-0.08, 0.12, -0.15, -0.33, 0.76, -0.11, -0.34, -0.25, 0.18, 0.71,
         0.61, -0.38, 0.92, 1.18, -0.17, -0.13, -0.07, 0.95, 0.71, 0.37])
    max_RAA = np.amax(RAA_table)
    min_RAA = np.amin(RAA_table)
    # print("max_RAA: ", max_RAA)
    # print("min_RAA: ", min_RAA)
    normolized_RAA_table = (RAA_table - min_RAA) / (max_RAA - min_RAA)
    # print("normalized_RAA_table: ", normolized_RAA_table)
    # normalized_RAA_table:
    # [0.19230769 0.32051282 0.1474359  0.03205128 0.73076923 0.17307692 0.02564103 0.08333333 0.35897436 0.69871795
    # 0.63461538 0. 0.83333333 1. 0.13461538 0.16025641 0.19871795 0.8525641 0.69871795 0.48076923]

    RAA_dict = {}
    RAA_dict['A'] = normolized_RAA_table[0]  # 0.19230769
    RAA_dict['R'] = normolized_RAA_table[1]  # 0.32051282
    RAA_dict['N'] = normolized_RAA_table[2]  # 0.1474359
    RAA_dict['D'] = normolized_RAA_table[3]  # 0.03205128
    RAA_dict['C'] = normolized_RAA_table[4]  # 0.73076923
    RAA_dict['Q'] = normolized_RAA_table[5]  # 0.17307692
    RAA_dict['E'] = normolized_RAA_table[6]  # 0.02564103
    RAA_dict['G'] = normolized_RAA_table[7]  # 0.08333333
    RAA_dict['H'] = normolized_RAA_table[8]  # 0.35897436
    RAA_dict['I'] = normolized_RAA_table[9]  # 0.69871795
    RAA_dict['L'] = normolized_RAA_table[10]  # 0.63461538
    RAA_dict['K'] = normolized_RAA_table[11]  # 0
    RAA_dict['M'] = normolized_RAA_table[12]  # 0.83333333
    RAA_dict['F'] = normolized_RAA_table[13]  # 1
    RAA_dict['P'] = normolized_RAA_table[14]  # 0.13461538
    RAA_dict['S'] = normolized_RAA_table[15]  # 0.16025641
    RAA_dict['T'] = normolized_RAA_table[16]  # 0.19871795
    RAA_dict['W'] = normolized_RAA_table[17]  # 0.8525641
    RAA_dict['Y'] = normolized_RAA_table[18]  # 0.69871795
    RAA_dict['V'] = normolized_RAA_table[19]  # 0.48076923

    return RAA_dict

def GetRAA(AA, RAA_dict):
    if (AA not in RAA_dict):
        print("[warning]: RAA_dict can't find ", AA, ". Returning 0")
        return 0
    else:
        return RAA_dict[AA]

def RetriveRAAFromASequence(seq, RAA_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    raa = []
    for index, item in enumerate(seq):
        raa.append(GetRAA(item, RAA_dict))
    return raa

def load_fasta_and_compute(seq_fn, out_fn, RAA_dict):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        raa = RetriveRAAFromASequence(line_Pseq, RAA_dict)
        fout.write(",".join(map(str,raa)) + "\n")
    fin.close()
    fout.close()

def main():
    RAA_dict = BuildRAADictionary()
    seq_fn = sys.argv[1]
    out_fn = sys.argv[2]
    load_fasta_and_compute(seq_fn, out_fn, RAA_dict)



if __name__ == '__main__':
    main()

