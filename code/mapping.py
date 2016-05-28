

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




#####################################
#  define PARAMETERS for learning:  #
#####################################
vects_type = 'maxpool' # type of representations used for LEARNING ('maxpool' or 'mean')
# good results with: model = 'linear', learningRate = 0.02, dropoutRate = 0.35, NumIterations = 5, hiddenUnits = 30
model = 'CCA' #OPTIONS: 'linear' or 'neuralnet' or 'softmax' or 'CCA'
learningRate = 0.02 # 0.02 is good for neuralnet. 0.0005 good for linear model.
dropoutRate = 0.25 # 0.25 - 0.35 work well for linear and neuralnet.
NumIterations = 10 #a hyperparameter that the library requires
#parameters JUST FOR NEURALNET:
hiddenUnits = 150 #just applies to neuralnet. 30 works quite well
activationFun = 'Tanh' #just applies to neuralnet. OPTIONS = ('Sigmoid' or 'Tanh' or 'Rectifier')
outputLayer = 'Linear' #just applies to neuralnet. OPTIONS =('Linear' or 'Softmax')


#Other parameters
test_data = False #to split or not (True/False) a partition of data for testing (10%by default). If False, evaluation is in training data
scale = False #scale the data
stored = False #if you want to import an existing model or LEARN a new one: True or False
store_model = True #if you want to store the model that you are using: True or False
map_embeddings = True
store_performance = True #if you want to store performance of these particular settings

savename = '_' + vects_type + '_' + model + '_LR_' + str(learningRate) + '_dropout_' + str(dropoutRate) + '_nhidden_' + str(hiddenUnits) + '_actiFun_' + activationFun



############################
# specify directories      #
############################
textDir = workingDir + '/TRAINING_DATA/wordvecs_all_imagenet.csv' #word embeddings dir
visualDir = workingDir + '/TRAINING_DATA/visual_vecs_all_imagenet_' + vects_type+ '.csv' #visual vectors dir
query_embeddDir = workingDir + '/embeddings/query_wordembeddings.csv' #query word embeddings
#query_wordsDir = workingDir + '/query_words.csv' #query words dir (OPTIONAL, just if names are not next to embeddings)
save_mappedDir = workingDir + '/embeddings/mapped_visual'+ savename +'.csv' #save the mapped output
save_modelDir = workingDir + '/models/MODEL' + savename + '.pkl' #save the mapped output
load_modelDir = save_modelDir
save_perfDir = workingDir + '/results/REGRESSION_' + savename + '.csv' #save the mapped output





############################
# READ data                #
############################
# 1. ########## WORD EMBEDDINGS and VISUAL vectors (i.e., input X and output y for the learning).
format = 'mixed' #choose 'numeric' (only numbers) OR 'mixed' if you have mixed data (first column is string)[word, vector_representation_dim_n]:

if format == 'numeric':
    #In NUMERIC ONLY format
    visual = np.loadtxt(open(visualDir,"rb"),delimiter=",",skiprows=0) # visual #READ csv into a numpy array
    text = np.loadtxt(open(textDir,"rb"),delimiter=",",skiprows=0)  # text #READ csv into a numpy array
if format == 'mixed':
    #OR if the input is in the MIXED format [word, vector_representation_dim_n]:
    import readDATA as rd
    words, visual = rd.readDATA(visualDir, format='csv') #get visual vectors
    _, text = rd.readDATA(textDir, format='csv') #get word embeddings


# 2. ######## QUERY data ###############
#get query words and their embedding: (OPTIONAL, just if you want to map something)
import readDATA as rd
query_words, embedding_query_words = rd.readDATA(query_embeddDir, format='csv')  # get visual vectors





################################
# IMPORT SAVED MODEL           # (OPTIONAL, instead of training)
################################
if stored == True:
    import pickle
    nn = pickle.load(open(load_modelDir, 'rb'))


######################
# TRAINING           # (skip that part if you already have a model to load)
######################

#train and test DATA
X_train = text
y_train = visual
#To test in actual test data (split of 10% for testing)
if test_data == True:
    X_train = text[0:np.floor(0.9*text.shape[0]), ]
    y_train = visual[0:np.floor(0.9*visual.shape[0]), ]
    X_test = text[np.floor(0.9*text.shape[0]):text.shape[0], ]
    y_test = visual[np.floor(0.9*visual.shape[0]):visual.shape[0], ]
else:
    X_test = X_train
    y_test = y_train

text = [] #empty memory
visual = [] #empty memory




#scale the data
if scale == True:
    #X = np.concatenate((X_train,X_test), axis=0)
    #X = pre.minmax_scale(X) #really bad results
    X = X_train #just in case we don't have a different test data
    X = pre.scale(X) #this scaling gives better results than minmax in regression
    X_train = X[0:X_train.shape[0],]
    X_test = X[X_train.shape[0]:X.shape[0],]
    X = [] #empty memory


#Define NEURAL NETWORK
if model == 'neuralnet':
    nn = Regressor(
        layers=[
            Layer(activationFun, units=hiddenUnits),
            Layer(outputLayer)],
        learning_rate=learningRate,
        n_iter=NumIterations, dropout_rate= dropoutRate)
    nn.fit(X_train, y_train)


#Define NEURAL NETWORK
if model == 'neuralnet2':
    nn = Regressor(
        layers=[
            Layer(activationFun, units=hiddenUnits),
            Layer(activationFun, units=100),
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


#define CCA
if model == 'CCA':
    from sklearn.cross_decomposition import CCA
    nn = CCA(copy=True, max_iter=500, n_components=1, scale=False, tol=1e-06)
    nn.fit(X_train, y_train)


#TODO: add autoencoder-pretrain


######################
# PREDICTION         #
######################
y_predicted = nn.predict(X_test)  # predict


#################
#  EVALUATION   # (to evaluate how well the REGRESSION did). For now we evaluate in the TRAINING DATA
#################
#TEST DATA
# R^2 measure
R2 = nn.score(X_test, y_test)
print('R^2_test= ',R2) #evaluating predictions with R^2
# My EVALUATION metric (mean cosine similarity)
cos = 0
for i in range(1,y_test.shape[0]):
    #cos = cos + np.dot(np.array(y_predicted[i,]), np.array(y_test[i,]))/ (np.linalg.norm(np.array(y_test[i,])) * np.linalg.norm(np.array(y_predicted[i,])))
    cos = cos + np.dot(y_predicted[i,], y_test[i,]) / (np.linalg.norm(y_test[i,]) * np.linalg.norm(y_predicted[i,]))
meanCos = cos/y_train.shape[0]
print('mean cos similarity_test= ', meanCos)

#TRAIN DATA:
y_predicted_train = nn.predict(X_train)  # predict
# R^2 measure
R2_train = nn.score(X_train, y_train)
print('R^2_train= ',R2_train) #evaluating predictions with R^2
# My EVALUATION metric (mean cosine similarity)
cos = 0
for i in range(1,y_train.shape[0]):
    #cos = cos + np.dot(np.array(y_predicted[i,]), np.array(y_test[i,]))/ (np.linalg.norm(np.array(y_test[i,])) * np.linalg.norm(np.array(y_predicted[i,])))
    cos = cos + np.dot(y_predicted_train[i,], y_train[i,]) / (np.linalg.norm(y_train[i,]) * np.linalg.norm(y_predicted_train[i,]))
meanCos_train = cos/y_train.shape[0]
print('mean cos similarity_train= ', meanCos_train)



#######################
#  STORE performance  # (optional)
#######################
if store_performance == True:
    if test_data == True:
        results = [ [ 'R^2_test' , R2 , 'R^2_train' , R2_train  ], ['mean cos simil_test', meanCos,'mean cos simil_train', meanCos_train] ]
    elif test_data == False:
        results = [ [ 'R^2_train' , R2_train  ], ['mean cos simil_train', meanCos_train] ]
    import writeDATA as wr
    wr.matrix2csv(results, save_perfDir)




#################
#  STORE MODEL  # (optional)
#################
if store_model == True:
    import pickle
    pickle.dump(nn, open(save_modelDir, 'wb'))


######################
#  MAP query words   # (optional) --> produces mapped visual representations
######################
if map_embeddings == True:
    mapped_query_words = nn.predict(embedding_query_words)  # predict



#WRITE mapped vectors (from query_words.csv)
import writeDATA as wr
wr.writeCSV(query_words, mapped_query_words, save_mappedDir)


X_train =[]
X_test = []
y_train = []
y_test = []






