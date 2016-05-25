
#given a list of words, it gets embeddings from GloVe, word2vec or wherever.

#get working directory
import os
print(os.getcwd() + "\n")
workingDir = os.getcwd()

#chose dirs:
wordembeddingDir = "/media/guillem/Untitled/GloVe/Wikipedia/glove.6B.50d_word2vec_format.txt" #Where GloVe or word2vec vectors are stored
wordsDir = workingDir + '/to_preprocess/unique_test_words.txt'
saveDir = workingDir + '/Outputs/query_embeddings.csv'
#wordembeddingDir = "/media/guillem/glove.840B.300d_word2vec_format.txt" #Where GloVe or word2vec vectors are stored
#wordsDir = workingDir + '/unique_test_words.txt'
#saveDir = workingDir + '/query_embeddings.csv'




#1. Get list of WORDS
words = open(wordsDir).read().splitlines()



#2. Get EMBEDDINGS for these words
from gensim.models import word2vec
#model = word2vec.Word2Vec.load_word2vec_format("/media/guillem/Untitled/GloVe/Wikipedia/glove.840B.300d_word2vec_format.txt", binary=False)
model = word2vec.Word2Vec.load_word2vec_format(wordembeddingDir, binary=False)
EMBEDDING = []
for i in range(len(words)):
    try:
        embedding = model[ words[i] ]  # NumPy vector of a word
        word = [ words[i] ]
        word.extend(embedding)
        EMBEDDING.append(word)
    except KeyError:
        print("word not found in your embedding vocabulary:", words[i])


# WRITE csv
import csv
with open(saveDir, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(EMBEDDING)










