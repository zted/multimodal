
#INPUT:  visual representations (and their corresponding words per row)

#PIPELINE: given a matrix with visual representations (first column is word),
# 1. we build first a matrix with the word embedding vectors
# 2. compute intersection between the two vocabularies.
# 3. then a concatenated vector (visual + text)

#BE CAREFUL: this code assumes there are NO REPEATED WORDS in the visual representations



#get working directory
import os
print(os.getcwd() + "\n")
workingDir = os.getcwd()


###################################################################
#  PRE-PROCESSING  (getting vectors + intersecting vocabularies)  #
###################################################################


# 1. Get words and VISUAL representations (just for rv_words.txt file, very special format)
visualDir = workingDir + '/Vectors/rv_words.txt'
#words_to_get = open(wordsDir).read().splitlines()
import numpy as np
f = open(visualDir, 'r')
visualVects=[]
words=[]
for line in f:
    line = line.strip()
    columns = line.split()
    name = columns[2]
    #to read Ted's vectors
    newstr = line.replace("[", "")
    newstr = newstr.replace("]", "")
    newstr = newstr.split("=")
    name = newstr[1]
    newstr = newstr[2]
    newstr = newstr.split(",")
    newstr =[float(i) for i in newstr]
    visualVects.append(newstr)
    words.append(name)

visualVects = np.array(visualVects)

#TODO: pre-processing step to get rid of repeated words


######## for the VGG_128 processing #############
import numpy as np
# 1. Get words
wordsDir = workingDir + '/Vectors/VGG_128/words.csv'
words = open(wordsDir).read().splitlines()
#1. Get VISUAL representations
visualDir = workingDir + '/Vectors/VGG_128/Avg.csv'
visualVects = np.loadtxt(open(visualDir,"rb"),delimiter=",",skiprows=0)





# 2. get WORD EMBEDDINGS of the relevant words (for which we have an image) assuming NOT REPEATED WORDS in above vocab!!
#'words' is the vector of words obtained from vision
relevantWords = []
relevantWordembeddings = []
with open("/media/guillem/Untitled/GloVe/glove.840B.300d.txt") as infile: #wherever you store the GloVe vectors
    for line in infile:
        word = line.split(" ")
        word = word[0]
        if word in words:
            relevantWordembeddings.append(line)
            relevantWords.append(word)

relevantWordembeddings
relevantWords

#TODO: write the GloVe vectors into word2vec format to allow the use of genism







#3. get INTERSECTION between visual vocabulary and word embedding vocabulary
#aa = set(relevantWords)
#bb = set(words)
#notFoundWords = bb - aa #we could do that since we are certain that list bb is larger than aa (because of our pipeline)
intersecVocab = list(set(relevantWords) & set(words))

# BE CAREFUL!!! list of words in word embeddings are in DIFFERENT ORDER than in VISUAL REPRESENTAIONS!!!
relevantWords #from word embeddings
words # from vision



######################
#  WRITING           #
######################

#4. write the TEXT ONLY embedding
saveDir = workingDir + '/Vectors/word_embeddings.csv'
#first create a matrix that will be saved as csv afterwards
EMBEDDING = []
for i in range(len(intersecVocab)):
    indices = [k for k, x in enumerate(relevantWords) if x == intersecVocab[i] ] #find index(es) of word i in the word embedding matrix
    embedding = list(relevantWordembeddings[j] for j in indices)  # retrieve the vectors
    embedding = str(embedding[0]) # WARNING: for now we just pick the first one!
    embedding = embedding.split() #the line already contains the WORD as a first element!
    #embedding = embedding[1:len(embedding)] #(optional) to remove the word
    #embedding = [float(t) for t in embedding2] #convert to float (optional)
    # currentWord = [ intersecVocab[i] ]
    # currentWord.extend(embedding)
    EMBEDDING.append(embedding)

#write the CSV file
import csv
with open(saveDir, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(EMBEDDING)





#5. write the VISUAL-ONLY embedding
saveDir = workingDir + '/Vectors/visual_vecs.csv'
#first create a matrix that will be saved as csv afterwards
VISUALVECS = []
for i in range(len(intersecVocab)):
    indices = [k for k, x in enumerate(words) if x == intersecVocab[i] ] #find index(es) of word i in the visual words list
    visualvec = visualVects[ indices[0] ,]  # WARNING: for now we just pick the first one!
    currentWord = [ intersecVocab[i] ]
    currentWord.extend(visualvec)
    VISUALVECS.append(currentWord)

#write the CSV file
import csv
with open(saveDir, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(VISUALVECS)





#6. write the MULTIMODAL embedding (concatenated)
saveDir = workingDir + '/Vectors/multimodal_embedding.csv'
gg = np.array(VISUALVECS)
gg = gg[:,1:gg.shape[1]] #remove words column
MULTIMODAL = np.concatenate((EMBEDDING,gg), axis=1) #concatenate by columns. EMBEDDING has the words
#write the CSV file
import csv
with open(saveDir, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(MULTIMODAL)




#7. write words.csv (for indexing) in order
saveDir = workingDir + '/Vectors/words.csv'
text_file = open(saveDir, "w")
for i in range(len(intersecVocab)):
    write_line = intersecVocab[i] + "\n"
    text_file.write(write_line)
text_file.close()

