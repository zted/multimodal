
###############################################
######## get statistics of entropy, std, etc. #
###############################################

import numpy as np
from sknn.mlp import Regressor, Layer
import sklearn.preprocessing as pre
from sklearn.datasets import samples_generator

#get working directory
import os
print(os.getcwd() + "\n")
workingDir = os.getcwd()

#add a path to the code (for the dependencies of import)
codeDir = workingDir + '/code'
import sys
sys.path.append(codeDir)

import readDATA as rd
word_attr_dict = rd.load_word_objects(workingDir + '/data/wordattributes.txt')

#get list of words test set
words , _ = rd.readDATA(workingDir + '/embeddings/query_visual_embeddings_mean.txt', 'spaces')

#words = words[1:len(words)]

entropies = []
stds = []
disps = []
concrs = []
for word in words:
    w_obj = word_attr_dict[word]
    entr = [w_obj.getAttribute('entropy')]
    std = [w_obj.getAttribute('std')]
    disp = [w_obj.getAttribute('dispersion')]
    concr = [w_obj.getAttribute('concreteness')]
    entropies.append(entr)
    stds.append(std)
    disps.append(disp)
    concrs.append(concr)


#flatten lists
concrs = [item for sublist in concrs for item in sublist] # flatten list
entropies = [item for sublist in entropies for item in sublist] # flatten list
stds = [item for sublist in stds for item in sublist] # flatten list
disps = [item for sublist in disps for item in sublist] # flatten list


#write a CSV file that with STRING elements
a = np.asarray([ ['---', 'entropy', 'std','dispersion','concreteness'], ['max',max(entropies),max(stds),max(disps),max(concrs)],
                 ['min',min(entropies),min(stds),min(disps),min([x for x in concrs if x is not None])],
                 ['median',np.median(entropies),np.median(stds),np.median(disps),np.median(concrs)] ])

import csv
with open(workingDir + '/data/descript_stats_disp.csv', "wb") as f:
    writer = csv.writer(f)
    writer.writerows(a)


from __future__ import division

mini = 2
maxi = 5

ccs = np.arange(mini, maxi +(maxi-mini)/10, (maxi-mini)/10)

for cc in ccs:
    print('foo', cc)

for i in range(len(ccs)):
    print('foo', ccs[i])

ccs[1]

maxi-mini

float()

(maxi-mini)/10


kk = [None]*len(ccs)


kk[1] = 'foo'

kk[3] = 65

kk

