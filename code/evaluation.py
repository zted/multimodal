

import numpy as np

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


def compute_similarity(v1, v2):
    v1 = [float(t) for t in v1]
    v2 = [float(t) for t in v2]
    cossim = 10 * np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )
    return cossim



################
# METHODS      #
################
# INPUT: tuples, wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
# OUTPUT: a vector of predictions (NUMERICAL)

def mapping_method(tuples, wordEmbeddings, mappedVisual, Visual , word_obj_dict, cutoff, measure):
    # INPUT: wordpairs,  wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
    # OUTPUT: a vector of predictions (NUMERICAL)
    #word_obj_dict: it is the matrix with entropy, disperson, etc.
    predictions = []
    for w1, w2 in tuples:
        WE1 = wordEmbeddings[w1]
        WE2 = wordEmbeddings[w2]
        w1_obj = word_obj_dict[w1]
        w2_obj = word_obj_dict[w2]

        w1_attribute_value = w1_obj.getAttribute(measure)
        w2_attribute_value = w2_obj.getAttribute(measure)

        if w1_attribute_value == None:
            # Does not have this attribute, do something, like use mapped
            VE1 = mappedVisual[w1]
        elif w1_attribute_value > cutoff:
            # Do something, maybe used mapped?
            VE1 = mappedVisual[w1]
            pass
        else:
            # Do something, use visual?
            VE1 = Visual[w1]
            pass

        if w2_attribute_value == None:
            # Does not have this attribute, do something, like use mapped
            VE2 = mappedVisual[w2]
        elif w2_attribute_value > cutoff:
            # Do something, maybe used mapped?
            VE2 = mappedVisual[w2]
            pass
        else:
            # Do something, use visual?
            VE2 = Visual[w2]
            pass
        #VE2 = np.array([0]*300) # <- placeholder, delete this later


        # concatenate the visual and word embeddings
        combined1 = np.concatenate([WE1, VE1], axis=0)
        combined2 = np.concatenate([WE2, VE2], axis=0)

        # compute similarity based on the combined embeddings
        sim_score = compute_similarity(combined1, combined2)
        predictions.append(sim_score)
    return predictions


def multimodal_always(tuples, wordEmbeddings, Visual): #concatenated always
    pass


def text_only_method(tuples, wordEmbeddings):
    pass


def visual_only_method(tuples, Visual): #uses all (and ONLY) visual representations (also meaningless)
    pass


#OPTIONAL
def visual_only_cutoff_method(tuples, mappedVisual, Visual, disp_value):  # uses some visual representations (dispersion cutoff) and maps the rest
    pass


if __name__ == "__main__":
    # get working directory
    import os
    print(os.getcwd() + "\n")
    workingDir = os.getcwd()
    # add a path to the code (for the dependencies of import)
    #codeDir = workingDir + '/code'
    #import sys
    #sys.path.append(codeDir)

    #some HYPERPARAMETERS:
    print(workingDir)
    measure = 'entropy'
    vects_type = 'maxpool'

    # load DATASET
    import readDATA as rd
    #datasetDir = workingDir + '../data/men.txt'
    datasetDir = '../data/men.txt'
    tuples, scores = rd.load_test_file(datasetDir)
    word_attr_dict = rd.load_word_objects('../data/wordattributes.txt')
    #set DIRECTORIES
    word_embDir = '../embeddings/query_wordembeddings.csv'
    mapped_vis_embDir =  '../embeddings/mapped_visual_' + vects_type + '.txt'
    vis_embDir = '../embeddings/query_visual_embeddings_' + vects_type + '.txt'
    #LOAD them
    word_emb = rd.load_embeddings(word_embDir) # some function to load embeddings
    mapped_vis_emb = rd.load_embeddings(mapped_vis_embDir) # some function to load embeddings
    vis_emb = rd.load_embeddings(vis_embDir) # some function to load embeddings

    # get DESCRIPTIVE STATISTICS of entropy, etc. (i.e., measure)
    #get words
    minim, maxim, med = rd.get_stats(vis_embDir, word_attr_dict, measure)

    #TODO: do a for loop to compute at different cutoffs
    cutoff_value = 1

    #PREDICT: method 1, method 2, etc.
    pred_map_method = mapping_method(tuples, word_emb, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)

    #EVALUATE PERFORMANCE: method 1, method 2, etc.
    #call perf measure
    performance = performance_measure(pred_map_method, scores)

    print(performance)

