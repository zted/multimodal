

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
        try: #if we don't even have wordembeddings for sure we won't have visual repr
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


def visual_only_map_allornone(tuples, mappedVisual, Visual , word_obj_dict, cutoff, measure):
    # mapping ALL OR NONE (only maps if both need to be mapped, e.g., both are above cutoff)
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
                if w1_attribute_value > cutoff and w2_attribute_value > cutoff:
                    VE1 = mappedVisual[w1]# Use mapped representation
                    VE2 = mappedVisual[w2]  # Use mapped representation
                else:
                    VE1 = Visual[w1]# Use orginal one
                    VE2 = Visual[w2]  # Use orginal one
                # compute similarity based on the combined embeddings
                sim_score = compute_similarity(VE1, VE2)
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



def visual_only_onlymap(tuples, mappedVisual):
    predictions = []
    for w1, w2 in tuples:
        try:
            VE2 = mappedVisual[w2]
            VE1 = mappedVisual[w1]
            sim_score = compute_similarity(VE1, VE2)
        except KeyError:
            sim_score = None
        predictions.append(sim_score)
    return predictions


def visual_only_nomap(tuples, Visual ):
    #NOTICE: this method maps less instances than visual_only_onlymap method since we have less images than wordembeddings
    #and onlymap method maps all embeddings.
    predictions = []
    for w1, w2 in tuples:
        try:
            VE2 = Visual[w2]
            VE1 = Visual[w1]
            sim_score = compute_similarity(VE1, VE2)
        except KeyError:
            sim_score = None
        predictions.append(sim_score)
    return predictions




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
    visual_vecs_type = 'maxpool' #VISUAL embeddings (choose 'mean' or 'maxpool')
    n_cutoffs = 10
    map_type = 'neuralnet' # choose: 'linear', 'neuralnet' , 'Softmax' or 'CCA' (BE CAREFUL. This must match mapped_vis_embDir)
    datasets = ['wordsim353','men','sensim_reduced','simlex999']
    #dataset = 'men'
    #dataset = 'wordsim353'
    #dataset = 'sensim_reduced'
    #dataset = 'simlex999'

    #Load EMBEDDINGS
    import readDATA as rd
    word_attr_dict = rd.load_word_objects('../data/wordattributes.txt')
    #set DIRECTORIES
    word_embDir = '../embeddings/query_wordembeddings.csv'
    #mapped_vis_embDir =  '../embeddings/mapped_visual_mean_linear_LR_0.002_dropout_0.2.csv'
    mapped_vis_embDir =  '../embeddings/mapped_visual_maxpool_neuralnet_LR_0.02_dropout_0.25_nhidden_150_actiFun_Tanh.csv'
    #mapped_vis_embDir =  '../embeddings/mapped_visual_maxpool_linear_LR_0.002_dropout_0.2.csv'
    vis_embDir = '../embeddings/query_visual_embeddings_' + visual_vecs_type + '.txt'
    #resultsDir = '../results/DATASET_' + dataset + '_' + visual_vecs_type + '_' + measure + '_' + map_type+'.csv'
    resultsDir = '../results/PERFORMANCE_' + visual_vecs_type + '_' + measure + '_' + map_type + '.csv'
    #LOAD them
    word_emb = rd.load_embeddings(word_embDir, 'csv') # some function to load embeddings
    mapped_vis_emb = rd.load_embeddings(mapped_vis_embDir, 'csv') # some function to load embeddings
    vis_emb = rd.load_embeddings(vis_embDir, 'spaces') # some function to load embeddings
    # get DESCRIPTIVE STATISTICS of entropy, etc. (i.e., measure)
    words, _ = rd.readDATA(vis_embDir, 'spaces') #get words
    minim, maxim, med = rd.get_stats(words, word_attr_dict, measure) #get descr stats for the measure of interest
    words = []

    text_file = open(resultsDir, "w")  # write RESULTS

    #datasets loop
    for dataset in datasets:
        print('dataset=', dataset)
        # load DATASET
        #datasetDir = workingDir + '../data/men.txt'
        datasetDir = '../data/' + dataset + '.txt'
        tuples, scores = rd.load_test_file(datasetDir)

        cutoffs = np.arange(minim , maxim + float(maxim-minim)/n_cutoffs, float(maxim-minim)/n_cutoffs) #generate cutoff values
        perf_map_method = [None]*len(cutoffs)
        perf_visual_only = [None] * len(cutoffs)
        perf_visual_only_allornone = [None] * len(cutoffs)
        for i in range(len(cutoffs)):
            cutoff_value = cutoffs[i]
            #PREDICT: method 1, method 2, etc.
            pred_map_method = mapping_method(tuples, word_emb, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
            pred_visual_only = visual_only_map(tuples, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
            pred_visual_only_allornone = visual_only_map_allornone(tuples, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value, measure)
            #EVALUATE PERFORMANCE: method 1, method 2, etc.
            perf_map_method[i] = performance_measure(pred_map_method, scores)
            perf_visual_only[i] = performance_measure(pred_visual_only, scores)
            perf_visual_only_allornone[i] = performance_measure(pred_visual_only_allornone, scores)
        #TEXT ONLY method
        pred_text_only = text_only(tuples, word_emb)
        perf_text_only = performance_measure(pred_text_only, scores)
        #RANDOM guessing method
        pred_random = random_method(tuples)
        perf_random = performance_measure(pred_random,scores)
        #ONLY MAP (and just visual)
        pred_visual_only_onlymap = visual_only_onlymap(tuples,mapped_vis_emb)
        perf_visual_only_onlymap = performance_measure(pred_visual_only_onlymap,scores)
        #ONLY visual and not mapped (just when we have visual embeddings!)
        pred_visual_only_nomap = visual_only_nomap(tuples, vis_emb)
        perf_visual_only_nomap = performance_measure(pred_visual_only_nomap,scores)

        print('visual_only_allornone=' ,[round(el, 2) for el in perf_visual_only_allornone ])
        print('visual_only_map=' ,[round(el, 2) for el in perf_visual_only ])
        print('mapping_multim_method=' , [round(el, 2) for el in perf_map_method ])
        print('text_only=', round(perf_text_only, 2) )
        print('random_method=', round(perf_random, 2) )
        print('visual_only_allmapped=', round(perf_visual_only_onlymap, 2))
        print('visual_only_nomap', round(perf_visual_only_nomap, 2))


        # text_file = open('C:/Guillem(work)/KU_Leuven/Code/VGG_128/syns_and_words.txt', "w")
        text_file.write('########### ' + dataset + "######## \n")
        text_file.write('### ' + mapped_vis_embDir + "### \n")
        text_file.write('visual_only_allornone=' + "," + ",".join([str(round(el,2)) for el in perf_visual_only_allornone ]) + "\n")
        text_file.write('visual_only_map=' + "," + ",".join([str(round(el,2)) for el in perf_visual_only]) + "\n")
        text_file.write('mapping_multim_method=' + "," + ",".join([str(round(el,2)) for el in perf_map_method]) + "\n")
        text_file.write('text_only=' + "," + str(round(perf_text_only,2) ) + "\n")
        text_file.write('random_method=' + "," + str(round(perf_random,2)) + "\n")
        text_file.write('visual_only_allmapped=' + "," + str(round(perf_visual_only_onlymap,2)) + "\n")
        text_file.write('visual_only_nomap=' + "," + str(round(perf_visual_only_nomap,2)) + "\n")


    text_file.close()




