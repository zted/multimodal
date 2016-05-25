

import numpy as np
from sknn.mlp import Regressor, Layer
import sklearn.preprocessing as pre
from sklearn.datasets import samples_generator

#get working directory
import os
print(os.getcwd() + "\n")
workingDir = os.getcwd()


############################
# specify directories      #
############################

# 1. ########## TEXT and VISUAL vectors (i.e., input X and output y for the learning)
# visual
visualDir = workingDir + '/VGG_128_SRL/visual_vecs_Maxpool.csv'
visual = np.loadtxt(open(visualDir,"rb"),delimiter=",",skiprows=0) #READ csv into a numpy array
# text
textDir = workingDir + '/VGG_128_SRL/word_embeddings_Maxpool.csv'
text = np.loadtxt(open(textDir,"rb"),delimiter=",",skiprows=0) #READ csv into a numpy array

######## query words ###############
#get query words and their embedding: (OPTIONAL, just if you want to map something)
#TODO: get word embeddings for the words that are in query_words.csv but we do not have visual representation for
#words
query_wordsDir = workingDir + '/query_words.csv'
#query_words =
#word embeddings
query_embeddDir = workingDir + '/embedding_query_words.csv'
#embedding_query_words =
#save the mapped output:
save_mappedDir = workingDir + '/mapped_visual_repr.txt'



#####################################
#  define PARAMETERS for learning:  #
#####################################
# good results with: model = 'linear', learningRate = 0.02, dropoutRate = 0.35, NumIterations = 5, hiddenUnits = 30
scale = False #scale the data
model = 'neuralnet' #OPTIONS: 'linear' or 'neuralnet' or 'softmax' or 'CCA'
learningRate = 0.02 # 0.02 is good for neuralnet. 0.0005 good for linear model.
dropoutRate = 0.25 # 0.25 - 0.35 work well for linear and neuralnet.
NumIterations = 5 #a hyperparameter that the library requires
#parameters JUST FOR NEURALNET:
hiddenUnits = 50 #just applies to neuralnet. 30 works quite well
activationFun = 'Tanh' #just applies to neuralnet. OPTIONS = ('Sigmoid' or 'Tanh' or 'Rectifier')
outputLayer = 'Linear' #just applies to neuralnet. OPTIONS =('Linear' or 'Softmax')


################################
# IMPORT SAVED MODEL           # (OPTIONAL, instead of training)
################################
import pickle
nn = pickle.load(open('nn.pkl', 'rb'))


######################
# TRAINING           # (skip that part if you already have a model to load)
######################

#train and test DATA
X_train = text
y_train = visual
#for now we test in the training data
X_test = text
y_test = visual


#scale the data
if scale == True:
    X = np.concatenate((X_train,X_test), axis=0)
    #X = pre.minmax_scale(X) #really bad results
    X = pre.scale(X) #this scaling gives better results than minmax in regression
    X_train = X[0:X_train.shape[0],]
    X_test = X[X_train.shape[0]:X.shape[0],]


#Define NEURAL NETWORK
if model == 'neuralnet':
    nn = Regressor(
        layers=[
            Layer(activationFun, units=hiddenUnits),
            Layer(outputLayer)],
        learning_rate=learningRate,
        n_iter=NumIterations, dropout_rate= dropoutRate)
    nn.fit(X_train, y_train)


#Define LINEAR Model
if model == 'linear':
    nn = Regressor(
        layers=[ Layer("Linear")],
        learning_rate=learningRate, #very small learning rate works better for the linear!
        n_iter=NumIterations, dropout_rate= dropoutRate)
    nn.fit(X_train, y_train)


#Define Softmax model (just a logistic classifier)
if model == 'softmax':
    nn = Regressor(
        layers=[ Layer("Softmax")],
        learning_rate=learningRate, #very small learning rate works better for the linear!
        n_iter=NumIterations, dropout_rate= dropoutRate)
    nn.fit(X_train, y_train)

#TODO: add CCA model and autoencoder-pretrain


######################
# PREDICTION         #
######################
y_predicted = nn.predict(X_test)  # predict


#################
#  EVALUATION   # (to evaluate how well the REGRESSION did)
#################
# R^2 measure
print(nn.score(X_test,y_test)) #evaluating predictions with R^2

# My EVALUATION metric (mean cosine similarity)
cos = 0
for i in range(1,y_test.shape[0]):
    #cos = cos + np.dot(np.array(y_predicted[i,]), np.array(y_test[i,]))/ (np.linalg.norm(np.array(y_test[i,])) * np.linalg.norm(np.array(y_predicted[i,])))
    cos = cos + np.dot(y_predicted[i,], y_test[i,]) / (np.linalg.norm(y_test[i,]) * np.linalg.norm(y_predicted[i,]))
meanCos = cos/y_test.shape[0]
print(meanCos)


#################
#  STORE MODEL  # (optional)
#################
import pickle
pickle.dump(nn, open('nn.pkl', 'wb'))



######################
#  MAP query words   # (optional) --> produces mapped visual representations
######################
mapped_query_words = nn.predict(embedding_query_words)  # predict



#WRITE mapped vectors (from query_words.csv)
import writeDATA as wr
wr.writeCSV(query_words,mapped_query_words, save_mappedDir)








