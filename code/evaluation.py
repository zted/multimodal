

import numpy as np

def performance_measure(predicted, groundtruth):
    #INPUT: predicted vector and groundtruth (both numerical)
    #OUTPUT: the right metric
    import numpy as np
    import scipy.stats
    indices = [i for i, x in enumerate(predicted) if x != None] # get rid of None values
    groundtruth = [groundtruth[i] for i in indices]
    predicted = [predicted[i] for i in indices]
    predicted = np.array(predicted)
    groundtruth = np.array(groundtruth)
    #perf = ((predicted - groundtruth) ** 2).mean(axis=None)
    perf = scipy.stats.spearmanr(groundtruth, predicted)[0]
    return perf


def compute_similarity(v1, v2):
    v1 = [float(t) for t in v1]
    v2 = [float(t) for t in v2]
    cossim = 10 * np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )
    return cossim



################
# METHODS      #
################
# INPUT: wordpairs,  wordEmbeddings, mappedVisual, Visual , disp_matrix, disp_cutoff
# OUTPUT: a vector of predictions (NUMERICAL)
# word_obj_dict: it is the matrix with entropy, disperson, etc.


def mapping_method(tuples, wordEmbeddings, mappedVisual, Visual , word_obj_dict, cutoff, measure):
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
                combined1 = WE1
                combined2 = WE2
            else: #if BOTH words have a visual embedding
                #first word
                if w1_attribute_value > cutoff:
                    VE1 = mappedVisual[w1]# Use mapped representation
                    pass
                else:
                    VE1 = Visual[w1]# Use orginal one
                    pass
                #second word:
                if w2_attribute_value > cutoff:
                    VE2 = mappedVisual[w2]# Use mapped representation
                    pass
                else:
                    VE2 = Visual[w2]# Use orginal one
                    pass
                # concatenate the visual and word embeddings
                combined1 = np.concatenate([WE1, VE1], axis=0)
                combined2 = np.concatenate([WE2, VE2], axis=0)

            sim_score = compute_similarity(combined1, combined2)# compute similarity based on the combined embeddings
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


def visual_only_map(tuples, mappedVisual, Visual , word_obj_dict, cutoff, measure):
    predictions = []
    for w1, w2 in tuples:
        try:
            w1_obj = word_obj_dict[w1]
            w2_obj = word_obj_dict[w2]

            w1_attribute_value = w1_obj.getAttribute(measure)
            w2_attribute_value = w2_obj.getAttribute(measure)

            if w1_attribute_value == None or w2_attribute_value == None: #if at least ONE doesn't have visual emb, use wordembeddings
                sim_score = None # just predict when it is visual
            else: #if BOTH words have a visual embedding
                #first word
                if w1_attribute_value > cutoff:
                    VE1 = mappedVisual[w1]# Use mapped representation
                    pass
                else:
                    VE1 = Visual[w1]# Use orginal one
                    pass
                #second word:
                if w2_attribute_value > cutoff:
                    VE2 = mappedVisual[w2]# Use mapped representation
                    pass
                else:
                    VE2 = Visual[w2]# Use orginal one
                    pass
                # concatenate the visual and word embeddings
                combined1 = VE1
                combined2 = VE2
                # compute similarity based on the combined embeddings
                sim_score = compute_similarity(combined1, combined2)
        except KeyError:
            sim_score = None

        predictions.append(sim_score)
    return predictions


def random_method(tuples):
    predictions = []
    for w1, w2 in tuples:
            WE1 = np.random.uniform(0,1,300)
            WE2 = np.random.uniform(0,1,300)
            sim_score = compute_similarity(WE1, WE2)
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
    measure = 'entropy'
    vects_type = 'mean'
    n_cutoffs = 10
    #dataset = 'men'
    dataset = 'wordsim353'
    #dataset = 'sensim_reduced'
    #dataset = 'simlex999'


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
    perf_visual_only = [None] * len(cutoffs)
    for i in range(len(cutoffs)):
        cutoff_value = cutoffs[i]
        #PREDICT: method 1, method 2, etc.
        pred_map_method = mapping_method(tuples, word_emb, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
        pred_visual_only = visual_only_map(tuples, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
        #EVALUATE PERFORMANCE: method 1, method 2, etc.
        perf_map_method[i] = performance_measure(pred_map_method, scores)
        perf_visual_only[i] = performance_measure(pred_visual_only, scores)
        print('performance mapping method=', perf_map_method[i])
    #TEXT ONLY method
    pred_text_only = text_only(tuples, word_emb)
    perf_text_only = performance_measure(pred_text_only, scores)
    #RANDOM guessing method
    pred_random = random_method(tuples)
    perf_random = performance_measure(pred_random,scores)

    print('visual_only=' ,perf_visual_only)
    print('mapping_method=' ,perf_map_method)
    print('text_only', perf_text_only)
    print('random_method', perf_random)




