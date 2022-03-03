#!/bin/csh
#$ -q gpu@@csecri
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y

/afs/crc.nd.edu/user/n/nsapkota/miniconda3/envs/psu-tf1.15/bin/python  test.py 
