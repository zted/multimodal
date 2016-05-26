import numpy as np
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

def compute_similarity(v1, v2):
    # TODO: fill this in
    return 0


def mapping_method(tuples, wordEmbeddings, mappedVisual, Visual , word_obj_dict, cutoff):
    # INPUT: wordpairs, dispersion
    # OUTPUT: a vector of predictions (NUMERICAL)
    predictions = []
    for w1, w2 in tuples:
        WE1 = wordEmbeddings[w1]
        WE2 = wordEmbeddings[w2]
        w1_obj = word_obj_dict[w1]
        w2_obj = word_obj_dict[w2]

        w1_attribute_value = w1_obj.getAttribute('entropy')
        w2_attribute_value = w2_obj.getAttribute('entropy')

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

        # TODO: do the same thing for the second word
        VE2 = np.array([0]*50) # <- placeholder, delete this later

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

    #evaluates the methods at each dispersion value (just mapping method changes)
    import readDATA as rd
    datasetDir = 'path_to_dataset'
    tuples, scores = rd.load_test_file(datasetDir)

    word_attr_dict = rd.load_word_objects('../data/wordattributes.txt')
    word_emb = 'path_to_embeddings'
    mapped_vis_emb = 'path_to_embeddings'
    vis_emb = 'path_to_embeddings'
    word_emb = rd.load_embeddings(word_emb) # some function to load embeddings
    mapped_vis_emb = rd.load_embeddings(mapped_vis_emb) # some function to load embeddings
    vis_emb = rd.load_embeddings(vis_emb) # some function to load embeddings
    cutoff_value = 1

    #PREDICT: method 1, method 2, etc.

    prediction_vectors = mapping_method(tuples, word_emb, mapped_vis_emb, vis_emb, word_attr_dict, cutoff_value)
    #call methods
    #EVALUATE PERFORMANCE: method 1, method 2, etc.
    #call perf measure

