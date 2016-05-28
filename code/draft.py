
import numpy as np
#get working directory
import os
print(os.getcwd() + "\n")
workingDir = os.getcwd()
#add a path to the code (for the dependencies of import)
codeDir = workingDir + '/code'
import sys
sys.path.append(codeDir)

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


####################################
# check visual vecs actually work  #
####################################

#get list of words test set
import readDATA as rd
#words, vecs = rd.readDATA(workingDir + '/embeddings/query_visual_embeddings_mean.txt', 'spaces')
words, vecs = rd.readDATA(workingDir + '/embeddings/mapped_visual_maxpool_neuralnet_LR_0.02_dropout_0.25_nhidden_150_actiFun_Tanh.csv', 'csv')

vecs[7,]
buddy = vecs[7,]
captain = vecs[4,]
swan = vecs[8, ]
trousers = vecs[10,]
parrot = vecs[16,]
clothes = vecs[18,]
bicycle = vecs[20,]
bike = vecs[9,]
blouse = vecs[26,]
men = vecs[30,]
kids = vecs[33,]
sweater = vecs[38,]
president = vecs[66,]




def simil(v1,v2):
    v1 = [float(t) for t in v1]
    v2 = [float(t) for t in v2]
    cossim = 10 * np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(cossim)


print('president and captain', simil(president,captain))
print('buddy and captain', simil(buddy,captain))
print('president and bicycle', simil(president,bicycle))
print('bike and bicycle', simil(bike,bicycle))
print('swan and parrot', simil(swan,parrot))
print('blouse and parrot', simil(blouse,parrot))
print('blouse and sweater', simil(blouse,sweater))
print('clothes and sweater', simil(clothes,sweater))



############ now check the training data #####################
words, vecs = rd.readDATA(workingDir + '/TRAINING_DATA/visual_vecs_all_imagenet_maxpool.csv', 'csv')

#get indices of a list that contain a given element whatever
indices = [i for i, x in enumerate(words) if x == "bike"]
print(indices)
bike = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "bicycle"]
print(indices)
bicycle = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "buddy"]
print(indices)
buddy = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "captain"]
print(indices)
captain = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "swan"]
print(indices)
swan = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "president"]
print(indices)
president = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "sweater"]
print(indices)
sweater = vecs[indices[0],]
indices = [i for i, x in enumerate(words) if x == "clothes"]
print(indices)
clothes = vecs[indices[0],]

print('president and captain', simil(president,captain))
print('buddy and captain', simil(buddy,captain))
print('president and bicycle', simil(president,bicycle))
print('bike and bicycle', simil(bike,bicycle))
print('swan and parrot', simil(swan,parrot))
print('blouse and parrot', simil(blouse,parrot))
print('blouse and sweater', simil(blouse,sweater))
print('clothes and sweater', simil(clothes,sweater))


# see if the representations of a given word (different synsets) are consistent.
indices = [i for i, x in enumerate(words) if x == "bicycle"]
print(indices)

for i in range(len(indices)):
    for j in range(i):
        print(simil(vecs[indices[i],],vecs[indices[j],]) )




###########################################################
# snippet to generate word embeddings and word attributes #
###########################################################

outfile = 'data/visual_embeddings_mean.txt'
fo = open(outfile, 'w')
for word in testwords:
    retrieved = False
    syns = wn.synsets(word)
    if syns == []:
        continue
    for s in syns:
        offset = str(s.offset())
        try:
            mystr = ' '.join(mydict[offset])
            retrieved = True
            break
        except KeyError:
            pass
    if retrieved:
        fo.write('{} {}'.format(word, mystr))
fo.close()




#get indices of a list that contain a given element whatever
indices = [i for i, x in enumerate(my_list) if x == "whatever"]

#or to get indices if what you have is a numpy array (NUMERIC) instead of a list
import numpy as np
values = np.array([1,2,3,1,2,4,5,6,3,2,1])
np.where(values == 3)[0]
#or alternatively
(values == 3).nonzero()

#to use the indices we found before to get the elements in another vector/list
c2 = [6544, 6 ,76,8765, 3434]
cc = [6544, 6 ,None,8765, None]
indices = [i for i, x in enumerate(cc) if x == '3ds']
list( c2[i] for i in indices ) #the new list to retrieve is c2




vec0 = [1, 6 ,5, 8, 4, 9]
vec = [3, None ,5, None, 4, 1]
indices = [i for i, x in enumerate(vec) if x != None]
vec0 = [vec0[i] for i in indices]
vec0

vec = [x for x in vec if x is not None]

indices


import scipy.stats
a= [1,2,3,4,5]
b = [3,4,5,6,7]
perf = scipy.stats.spearmanr(b, a)[0]
perf
