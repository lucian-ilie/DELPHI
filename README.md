# DELPHI
Deep learning protein-protein binding sites prediction

## Citation
Li, Yiwei and Ilie, Lucian, DELPHI: accurate deep ensemble model for protein interaction sites prediction. bioRxiv.

##
Contact: 
Yiwei Li (yli922@uwo.ca)

Lucian Ilie (ilie@uwo.ca)
## Installation
The python (3.6) virtural enviroment wiht pip3 is highly recommended. All required python packages are listed in requirement'txt. The following pip command will install all of them at once.
```
pip install -r requirement.txt
```
## Datasets
All datasets used in the paper can be downloaded at https://www.csd.uwo.ca/~ilie/DELPHI/

## Directory explaination 
1. `Src` 
Src directory contains source files for training and predicting. All program options can be seen by typing 
```
python train.py
python predic.py
```
Two scripts `run_training.sh` and `run_predicting.sh` are included as an example of how to run the programs.

2. `Saved_Model`
Saved_model dir contains the keras model of the final ensemble model.

3. `Feature_Computation`
Feature_Computation contains programs and script piplines for computing each script. 

## DELPHI model architecture 
![](img/Model_architecture.jpg)

![](img/many_2_one.jpg)
