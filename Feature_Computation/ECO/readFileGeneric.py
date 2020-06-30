import sys, traceback
import pandas as pd
import os
import math
#from sklearn import preprocessing
import numpy as np
from scipy.stats import kstest
import random
from scipy import stats

# A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y
A =0.07800
C =0.02400
D =0.05200
E =0.05900
F =0.04400
G =0.08300
H =0.02500
I =0.06200
K =0.05600
L =0.09200
M =0.02400
N =0.04100
P =0.04300
Q =0.03400
R =0.05100
S =0.05900
T =0.05500
V =0.07200
W =0.01400
Y =0.03400
X =1
U =1

background  = [C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,A]


backgrounddict ={
'A' : 0.07800,
'C' : 0.02400,
'D' : 0.05200,
'E' : 0.05900,
'F' : 0.04400,
'G' : 0.08300,
'H' : 0.02500,
'I' : 0.06200,
'K' : 0.05600,
'L' : 0.09200,
'M' : 0.02400,
'N' : 0.04100,
'P' : 0.04300,
'Q' : 0.03400,
'R' : 0.05100,
'S' : 0.05900,
'T' : 0.05500,
'V' : 0.07200,
'W' : 0.01400,
'Y' : 0.03400,
'X' : 1,
'U' : 1
}

def readBlosumMatrix(path) :

    df = pd.read_csv(path,delimiter='\t', header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.rename(columns={'HMM    A':'A'}, inplace=True)
    df = df[pd.notna(df['Y'])]
    new = df["A"].str.split(' ', n=2, expand=True)
    df_concat = pd.concat([df, new], axis=1)
    df_concat = df_concat.drop(columns=['A'])
    df_concat = df_concat.drop(columns=[0, 1])
    df_concat.rename(columns={2:'A'}, inplace=True)
    # A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y
    df = df_concat
    df.columns = ['C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','A']
    #print('#####################')

    #print('Raw values Blosum ')
    df = df.reset_index(drop=True)
    #print(df)

    # A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y
    df['A'] = df['A'].astype(str)
    df['C'] = df['C'].astype(str)
    df['D'] = df['D'].astype(str)
    df['E'] = df['E'].astype(str)
    df['F'] = df['F'].astype(str)
    df['G'] = df['G'].astype(str)
    df['H'] = df['H'].astype(str)
    df['I'] = df['I'].astype(str)
    df['K'] = df['K'].astype(str)
    df['L'] = df['L'].astype(str)
    df['M'] = df['M'].astype(str)
    df['N'] = df['N'].astype(str)
    df['P'] = df['P'].astype(str)
    df['Q'] = df['Q'].astype(str)
    df['R'] = df['R'].astype(str)
    df['S'] = df['S'].astype(str)
    df['T'] = df['T'].astype(str)
    df['V'] = df['V'].astype(str)
    df['W'] = df['W'].astype(str)
    df['Y'] = df['Y'].astype(str)

    # A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y

    df.loc[df['A'].str.contains('*', regex=False), 'A'] = "999999999"
    df.loc[df['C'].str.contains('*', regex=False), 'C'] = "999999999"
    df.loc[df['D'].str.contains('*', regex=False), 'D'] = "999999999"
    df.loc[df['E'].str.contains('*', regex=False), 'E'] = "999999999"
    df.loc[df['F'].str.contains('*', regex=False), 'F'] = "999999999"
    df.loc[df['G'].str.contains('*', regex=False), 'G'] = "999999999"
    df.loc[df['H'].str.contains('*', regex=False), 'H'] = "999999999"
    df.loc[df['I'].str.contains('*', regex=False), 'I'] = "999999999"
    df.loc[df['K'].str.contains('*', regex=False), 'K'] = "999999999"
    df.loc[df['L'].str.contains('*', regex=False), 'L'] = "999999999"
    df.loc[df['M'].str.contains('*', regex=False), 'M'] = "999999999"
    df.loc[df['N'].str.contains('*', regex=False), 'N'] = "999999999"
    df.loc[df['P'].str.contains('*', regex=False), 'P'] = "999999999"
    df.loc[df['Q'].str.contains('*', regex=False), 'Q'] = "999999999"
    df.loc[df['R'].str.contains('*', regex=False), 'R'] = "999999999"
    df.loc[df['S'].str.contains('*', regex=False), 'S'] = "999999999"
    df.loc[df['T'].str.contains('*', regex=False), 'T'] = "999999999"
    df.loc[df['V'].str.contains('*', regex=False), 'V'] = "999999999"
    df.loc[df['W'].str.contains('*', regex=False), 'W'] = "999999999"
    df.loc[df['Y'].str.contains('*', regex=False), 'Y'] = "999999999"


    # A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y


    df['A'] = df['A'].astype(float)
    df['C'] = df['C'].astype(float)
    df['D'] = df['D'].astype(float)
    df['E'] = df['E'].astype(float)
    df['F'] = df['F'].astype(float)
    df['G'] = df['G'].astype(float)
    df['H'] = df['H'].astype(float)
    df['I'] = df['I'].astype(float)
    df['K'] = df['K'].astype(float)
    df['L'] = df['L'].astype(float)
    df['M'] = df['M'].astype(float)
    df['N'] = df['N'].astype(float)
    df['P'] = df['P'].astype(float)
    df['Q'] = df['Q'].astype(float)
    df['R'] = df['R'].astype(float)
    df['S'] = df['S'].astype(float)
    df['T'] = df['T'].astype(float)
    df['V'] = df['V'].astype(float)
    df['W'] = df['W'].astype(float)
    df['Y'] = df['Y'].astype(float)

    # A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y

    df['A'] = df['A'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['C'] = df['C'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['D'] = df['D'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['E'] = df['E'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['F'] = df['F'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['G'] = df['G'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['H'] = df['H'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['I'] = df['I'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['K'] = df['K'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['L'] = df['L'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['M'] = df['M'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['N'] = df['N'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['P'] = df['P'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['Q'] = df['Q'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['R'] = df['R'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['S'] = df['S'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['T'] = df['T'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['V'] = df['V'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['W'] = df['W'].apply(lambda x: float(2 ** (-float(x) / 1000)))
    df['Y'] = df['Y'].apply(lambda x: float(2 ** (-float(x) / 1000)))

    return df


def getAllFilesinOrder(hpath):
    valueFile = open(hpath,"r")
    allFiles = valueFile.readlines()
    listFiles=[]
    for x in allFiles:
        if (x.startswith('>')):
            x = x[1:].rstrip("\n")+".fasta"
            listFiles.append(x)
            #print(x)
    print('listFiles', listFiles)
    return listFiles


def calculateFeature(row,aa):
    featureVal =0
    featureValNr =0
    featureValDr =0
    x = aa
    val = backgrounddict[x]
    #print(' aa is ', x , ' val is ', val )
    #print('sum is ', np.sum(row))
    for i, c in enumerate(row):
        #print('ROW VALUES',i,c,background[i])
        featureValNr = featureValNr + (c*c)
        featureValDr = featureValDr + (c)

    #print('featureValNr',featureValNr)
    #print('featureValDr',featureValDr)

    featureValNr=featureValNr/val
    featureValDr=featureValDr/val

    featureVal = math.log(featureValNr,10)/math.log(featureValDr,10)
    #print("Feature", featureVal)
    return round(featureVal, 3)


def getHallMarkValues(fasta,path):
    hallMark = path+"/"+fasta
    f = open(hallMark, "r")
    for x in f:
        return x


def processFiles(path,fastaPath,mainFile,errorFile,featureOutFile) :
    count =0
    match =0
    errorFile = open(errorFile, "w+")
    featureOutFile = open(featureOutFile, "w+")
    featureValuesList = []
    horigpath = mainFile
    origOrder =  getAllFilesinOrder(horigpath)
    
    for filename in origOrder:
    #for filename in os.listdir(path):
        count = count+1
        calcValues = []
        hallMark =[]
        if filename.endswith(".fasta") :
            print('filename',filename)
            '''
            hpath = hallmarkPath+"/"+filename
            if (os.path.isfile(hpath)):
                hallMark = getHallMarkValues(filename,hallmarkPath)
                hallMark = [float(i) for i in hallMark.split()]
            else:
                errorFile.write('ERROR !!, Hallmark ' + filename+' DOESNT exist !!!!!!')
            '''
            #print('Hallmark',hallMark )
            print('Count ',count)
            fullPath =os.path.join(path, filename)
            fastaFilePath = os.path.join(fastaPath, filename)
            fasta = open(fastaFilePath)
            prot_name = fasta.readline()
            prot_seq = fasta.read().strip()
            if (count >1):
                featureOutFile.write('\n')
            featureOutFile.write(prot_name)
            featureOutFile.write(prot_seq)
            featureOutFile.write('\n')

            try:
                mat = readBlosumMatrix(fullPath)
                #print('Blosum')
                mat = mat.reset_index(drop=True)
                #print(mat)
                #print('Blosum dimension',mat.shape)
                #print(' protein length ', len(prot_seq) )
                #print('protein',prot_seq)
                line =""
                #print('Length', len(prot_seq))
                for i, aa in enumerate(prot_seq):
                    #print('enumerate',i,aa,mat.iloc[i])
                    feature = calculateFeature(mat.iloc[i],aa )
                    if (feature>0):
                        featureValuesList.append(feature)
                    else:
                        feature=0.0
                        featureValuesList.append(0.0)

                    if (len(line) > 0):
                        line = line + ","+ str(feature)
                    else:
                        line = str(feature)
                    calcValues.append(float(feature))
                    #print (i, ' ', sum(mat.iloc[i]))

                #print('feature List',featureValuesList)
                '''
                if (len(hallMark) >0):
                    len1 = len(calcValues)
                    len2 = len(hallMark)
                    if (len1==len2):
                        match = match+1
                        diff = [a_i - b_i for a_i, b_i in zip(hallMark, calcValues)]
                        diff=[abs(number) for number in diff]
                        mean = np.mean(diff)
                        std =  np.std(diff)
                        corr = np.corrcoef(hallMark, calcValues)[1][0]
                        #print('diff',mean , std)
                        #print('pear', corr)
                        #print(' len 1', len(calcValues),' len 2', len(hallMark))
                        meanList.append(mean)
                        stdList.append(std)
                        corrList.append(corr)
                        statsOutFile.write(filename+"\n")
                        statsOutFile.write("Mean "+str(mean)+"\n")
                        statsOutFile.write("Std "+str(std)+"\n")
                        statsOutFile.write("Correlation "+str(corr)+"\n")
                    else:
                        errorFile.write('ERRROR !! '+filename+' len of calcValues '+ str(len(calcValues))+' len of hallmark '+ str(len(hallMark))+"\n")
                else:
                    errorFile.write(filename+" doesnt exist in hallmark"+"\n")
                '''
                #print('diff',mean , std)
                #print('pear', corr)
                #print(' len 1', len(calcValues),' len 2', len(hallMark))

                #statsOutFile.write(filename+"\n")
                #statsOutFile.write("Mean "+str(mean)+"\n")
                #statsOutFile.write("Std "+str(std)+"\n")
                #statsOutFile.write("Correlation "+str(corr)+"\n")
                
                featureOutFile.write(line)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                traceback.print_exc(file=sys.stdout)
                print('error occured in file ',fullPath)

            continue
        else:
            continue
    '''
    mu = np.mean(meanList)
    sigma = np.mean(stdList)
    row = np.nanmean(corrList)
    statsOutFile.write('number of files '+ str(count)+"\n")
    statsOutFile.write('Final mean' + str(mu)+"\n")
    statsOutFile.write('Final Std'+str(sigma)+"\n")
    statsOutFile.write('Final Corr'+str(row)+"\n")
    statsOutFile.write('Matches '+str(match)+"\n")
    '''
    featureOutFile.close()
    #statsOutFile.close()
    errorFile.close()
def main():
    #out="/Users/sabby/Documents/ECO/out"

    #fastaPath= "/Users/sabby/Documents/ECO/filesBck"
    #fastaPath="/home/saby2k13/ECO/filesBck"

    fastaPath=sys.argv[1]
    #blosumPath="/Users/sabby/Documents/ECO/MSA"
    MSAPath = sys.argv[2]

    #hallmarkPath = "/Users/sabby/Documents/ECO/hallmarks/ECOFiles"
    #hallmarkPath = "/home/saby2k13/ECO/hallmarks/ECOFiles"

    errLoc = "errorlogfile.txt"
    fLoc = sys.argv[3]
    #statsLoc = out+"/stats.txt"

    print('errLoc',errLoc,'fLoc',fLoc)
    
    mainFile = sys.argv[4]
    processFiles(MSAPath,fastaPath,mainFile,errLoc,fLoc)

if __name__== "__main__":
    main()

