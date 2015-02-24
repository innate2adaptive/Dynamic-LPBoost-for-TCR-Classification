__author__ = 'yuxinsun'

import itertools
from scipy.io import savemat
from numpy import asarray
from time import clock
from collections import OrderedDict

path = 'INPUT YOUR PATH OF ORIGINAL DATA HERE'  # path of original data files
datafile = ['data1.txt', 'data2.txt', 'data3.txt']  # create a list of data file names
pathsave = 'INPUT YOUR PATH OF KERNEL MATRIX HERE'  # path of kernel matrix

p = 3  # spectrum
kern = []  # string kernel

# Build up triplets
alphabet = 'ACDEFGHIKLMNPQRSTVWY'

# Triplets
triplet = list(itertools.product(alphabet, repeat=p))
for i in range(0,len(triplet)):
    triplet[i] = ''.join((itertools.chain(triplet[i])))


for element in datafile:
    print('Start processing data...')
    data = []

    # Create dictionary
    dic = dict.fromkeys(triplet, 0)  # triplets
    dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))

    # Please modify the following code for varying formats of input data file
    # The code here is designed for input data such as
    # CASFGH
    # CASSYFQ
    # CASSADESCE
    # where each line contains one CDR3 sequence only

    # Read data files
    fileobject = open(path+element)
    filename = fileobject.readlines()
    filename = [s.strip('\r\n') for s in filename]

    for fileline in filename:
        s = fileline.split()
        data.append(s)

    # Join CDR3 sequences as a large text document
    ele = ' '.join((itertools.chain(data)))

    print('Start calculating kernel...')

    # Compute the p-spectrum string kernel with a Hash table
    for j in range(0, len(ele)-p+1):
        if ele[j:j+p] in dic:
            dic[ele[j:j+p]] += 1
    kern.append(dic.values())

# Save string kernel values to .mat file
a = asarray(kern)
savemat(pathsave+'stringkernel.mat', {'data':a}, oned_as='row')