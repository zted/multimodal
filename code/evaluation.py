
#Evaluate all the conditions in a data set:

def performance_measure(predicted, groundtruth):
    #INPUT: predicted vector and groundtruth (both numerical)
    #OUTPUT: the right metric
    #TODO: take care of NA values, which will occur each time we either don't have a vector
    # just eliminate NA values in a previous step
    import numpy as np
    predicted = np.array(predicted)
    groundtruth = np.array(groundtruth)
    perf = ((predicted - groundtruth) ** 2).mean(axis=None)
    return perf


################
# METHODS      #
################
# INPUT: tuples, wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
# OUTPUT: a vector of predictions (NUMERICAL)

def mapping_method(tuples, wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff):
    # INPUT: wordpairs, dispersion
    # OUTPUT: a vector of predictions (NUMERICAL)


def multimodal_always(tuples, wordEmbeddings, Visual): #concatenated always


def text_only_method(tuples, wordEmbeddings):


def visual_only_method(tuples, Visual): #uses all (and ONLY) visual representations (also meaningless)


#OPTIONAL
def visual_only_cutoff_method(tuples, mappedVisual, Visual, disp_value):  # uses some visual representations (dispersion cutoff) and maps the rest





def evaluate_dataset(datasetDir):
    #evaluates the methods at each dispersion value (just mapping method changes)
    import readDATA as rd
    tuples, scores = rd.load_test_file(datasetDir)
    #PREDICT: method 1, method 2, etc.
    #call methods
    #EVALUATE PERFORMANCE: method 1, method 2, etc.
    #call perf measure












