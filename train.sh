#!/bin/csh
#$ -q gpu@@csecri-v100  
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y

/afs/crc.nd.edu/user/n/nsapkota/miniconda3/envs/psu-tf1.15/bin/python  train.py -wb psu_ach_s165_d02272022_t1040p
