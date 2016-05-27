

import numpy as np

def performance_measure(predicted, groundtruth):
    #INPUT: predicted vector and groundtruth (both numerical)
    #OUTPUT: the right metric
    import numpy as np
    indices = [i for i, x in enumerate(predicted) if x != None] # get rid of None values
    groundtruth = [groundtruth[i] for i in indices]
    predicted = [predicted[i] for i in indices]
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

def mapping_method2(tuples, wordEmbeddings, mappedVisual, Visual , word_obj_dict, cutoff, measure):
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


def mapping_method(tuples, wordEmbeddings, mappedVisual, Visual , word_obj_dict, cutoff, measure):
    # If any of the two words is not visual, it uses word embeddings. If not, maps visual vecs when dispersion is larger than a cutoff.
    # INPUT: wordpairs,  wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
    # OUTPUT: a vector of predictions (NUMERICAL)
    #word_obj_dict: it is the matrix with entropy, disperson, etc.
    predictions = []
    for w1, w2 in tuples:
        try:
            WE1 = wordEmbeddings[w1]
            WE2 = wordEmbeddings[w2]
            w1_obj = word_obj_dict[w1]
            w2_obj = word_obj_dict[w2]

            w1_attribute_value = w1_obj.getAttribute(measure)
            w2_attribute_value = w2_obj.getAttribute(measure)

            if w1_attribute_value == None or w2_attribute_value == None: #if just ONE of them doesn't have visual emb, use wordembeddings
                # concatenate the visual and word embeddings
                combined1 = WE1
                combined2 = WE2
            else: #if BOTH words have a visual embedding
                #first word
                if w1_attribute_value > cutoff:
                    # Use mapped representation
                    VE1 = mappedVisual[w1]
                    pass
                else:
                    # Use orginal one
                    VE1 = Visual[w1]
                    pass
                #second word:
                if w2_attribute_value > cutoff:
                    # Use mapped representation
                    VE2 = mappedVisual[w2]
                    pass
                else:
                    # Use orginal one
                    VE2 = Visual[w2]
                    pass
                #VE2 = np.array([0]*300) # <- placeholder, delete this later
                # concatenate the visual and word embeddings
                combined1 = np.concatenate([WE1, VE1], axis=0)
                combined2 = np.concatenate([WE2, VE2], axis=0)

            # compute similarity based on the combined embeddings
            sim_score = compute_similarity(combined1, combined2)
        except KeyError:
            sim_score = None

        predictions.append(sim_score)
    return predictions


def text_only(tuples, wordEmbeddings):
    # INPUT: wordpairs,  wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
    # OUTPUT: a vector of predictions (NUMERICAL)
    #word_obj_dict: it is the matrix with entropy, disperson, etc.
    predictions = []
    for w1, w2 in tuples:
        try:
            WE1 = wordEmbeddings[w1]
            WE2 = wordEmbeddings[w2]
            # compute similarity based on the combined embeddings
            sim_score = compute_similarity(WE1,  WE2 )
        except KeyError:
            sim_score = None
        predictions.append(sim_score)
    return predictions

def visual_only(tuples, mappedVisual, Visual , word_obj_dict, cutoff, measure):
    #TODO: finish this method
    # INPUT: wordpairs,  wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
    # OUTPUT: a vector of predictions (NUMERICAL)
    #word_obj_dict: it is the matrix with entropy, disperson, etc.
    predictions = []
    for w1, w2 in tuples:
        try:
            WE1 = wordEmbeddings[w1]
            WE2 = wordEmbeddings[w2]
            # compute similarity based on the combined embeddings
            sim_score = compute_similarity(WE1,  WE2 )
        except KeyError:
            sim_score = None
        predictions.append(sim_score)
    return predictions


def multimodal_always(tuples, wordEmbeddings, Visual): #concatenated always
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
    measure = 'dispersion'
    vects_type = 'maxpool'
    n_cutoffs = 10
    dataset = 'wordsim353'
    #dataset = 'sensim_reduced'


    # load DATASET
    import readDATA as rd
    #datasetDir = workingDir + '../data/men.txt'
    datasetDir = '../data/' + dataset + '.txt'
    tuples, scores = rd.load_test_file(datasetDir)
    word_attr_dict = rd.load_word_objects('../data/wordattributes.txt')
    #set DIRECTORIES
    word_embDir = '../embeddings/query_wordembeddings.csv'
    mapped_vis_embDir =  '../embeddings/mapped_visual_maxpool_neuralnet_LR_0.02_dropout_0.25_nhidden_150_actiFun_Tanh.csv'
    vis_embDir = '../embeddings/query_visual_embeddings_' + vects_type + '.txt'
    #LOAD them
    word_emb = rd.load_embeddings(word_embDir, 'csv') # some function to load embeddings
    mapped_vis_emb = rd.load_embeddings(mapped_vis_embDir, 'csv') # some function to load embeddings
    vis_emb = rd.load_embeddings(vis_embDir, 'spaces') # some function to load embeddings

    # get DESCRIPTIVE STATISTICS of entropy, etc. (i.e., measure)
    words, _ = rd.readDATA(vis_embDir, 'spaces') #get words
    minim, maxim, med = rd.get_stats(words, word_attr_dict, measure) #get descr stats for the measure of interest
    words = []

    cutoffs = np.arange(minim, maxim + float(maxim-minim)/n_cutoffs, float(maxim-minim)/n_cutoffs) #generate cutoff values
    perf_map_method = [None]*len(cutoffs)
    for i in range(len(cutoffs)):
        cutoff_value = cutoffs[i]
        #cutoff_value = 1
        #PREDICT: method 1, method 2, etc.
        pred_map_method = mapping_method(tuples, word_emb, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
        #EVALUATE PERFORMANCE: method 1, method 2, etc.
        perf_map_method[i] = performance_measure(pred_map_method, scores)
    #tex only method
    pred_text_only = text_only(tuples, word_emb)
    perf_text_only = performance_measure(pred_text_only, scores)

    print('mapping_method=' ,perf_map_method)
    print('text_only', perf_text_only)




