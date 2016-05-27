

def readDATA(fileDir, format='csv'):
#Input: a csv or a space separated file with first column containing words and the rest numerical data
#Output: returns 1.numerical matrix with row vectors of words. 2.list of words (i.e., first column)
#format: either 'csv' or 'spaces'
    import numpy as np
    words = []
    vectors = []
    with open(fileDir) as infile:  # wherever you store the GloVe vectors
        for line in infile:
            line = line.strip()
            if format == 'csv':
                line = line.split(",")
            elif format == 'spaces':
                line = line.split(" ")
            word = line[0]
            words.append(word)
            vect = line[1:len(line)]  # to remove the word
            vect = [float(t) for t in vect]  # convert to float
            vectors.append(vect)
        vectors = np.array(vectors)
    return (words, vectors)



#load a test with
def load_test_file(fileDir):
    word_pairs = []
    scores = []
    with open(fileDir, 'r') as f:
        for line in f:
            splits = line.rstrip('\n').split('\t')
            word_pairs.append((splits[0].lower(), splits[1].lower()))
            scores.append(float(splits[2]))
    return word_pairs, scores


def load_embeddings(afile):
    """
    loads a csv file, returns dictionaries with words as keys,
    embeddings in the form of numpy arrays as values
    :param afile:
    :return:
    """
    import numpy as np
    embed_dict = {}
    with open(afile, 'r') as f:
        for line in f:
            splits = line.rstrip('\n').split(',')
            word = splits[0]
            embeddings = np.array(splits[1:])
            embed_dict[word] = embeddings
    return embed_dict


def load_word_objects(afile):
    """
    Loads words from a file into objects that contains the word's attributes.
    For example, the file should be formatted as such:
    WORD ENTROPY DISPERSION STD CONCRETENESS
    These values are read and initialized as objects, then passed back in
    a lookup table/dictionary
    :param afile: file containing words' attributes
    :return: dictionary where the key is a word, and its value is the object
    of the word
    """
    class WordObject(object):
        def __init__(self, name):
            self.name = name
            self.attributes = {'entropy': None,
                               'dispersion': None,
                               'std': None,
                               'concreteness': None}
            return

        def setAttribute(self, attribute, value):
            if value == 'N/A':
                pass
            else:
                self.attributes[attribute] = float(value)
            return

        def getAttribute(self, attribute):
            return self.attributes[attribute.lower()]

    word_dict = {}
    skipfirst = False
    # skips the first line since it is a header
    with open(afile, 'r') as f:
        for line in f:
            if skipfirst:
                skipfirst = True
                continue
            splits = line.rstrip('\n').split(' ')
            word = splits[0]
            entropy = splits[1]
            dispersion = splits[2]
            std = splits[3]
            concreteness = splits[4]
            wordObj = WordObject(word)
            wordObj.setAttribute('entropy', entropy)
            wordObj.setAttribute('dispersion', dispersion)
            wordObj.setAttribute('std', std)
            wordObj.setAttribute('concreteness', concreteness)
            word_dict[word] = wordObj
    return word_dict



def get_stats(vis_embDir, word_attr_dict, measure):
    import numpy as np
    import readDATA as rd
    #vis_embDir is the visual embeddings, to get the list of words
    # word_attr_dict: is the dictionary whith the entropy, etc.
    words, _ = rd.readDATA(vis_embDir, 'spaces')
    values = []
    for word in words:
        w_obj = word_attr_dict[word]
        value = [w_obj.getAttribute(measure)]
        values.append(value)
    values = [item for sublist in values for item in sublist]  #first flatten the list
    minim = min([x for x in values if x is not None])
    maxim = max(values)
    med = np.median(values)
    return(minim, maxim, med)
















