

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












