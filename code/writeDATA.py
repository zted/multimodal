#this code writes data in CSV

def writeCSV(words, matrix, saveDir):
    import csv
    #words: is a numpy array or an array
    #matrix: must be a NUMERICAL NUMPY ARRAY
    #1. build the "joint" matrix
    MATRIX = []
    for i in range(len(words)):
        vec = matrix[i,]  # WARNING: for now we just pick the first one!
        currentWord = [ words[i] ]
        currentWord.extend(vec)
        MATRIX.append(currentWord)
    # 2. write the CSV file
    with open(saveDir, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)
    #return MATRIX


#TODO: write to space-separated file

def matrix2csv(MATRIX, saveDir): #assumes MATRIX is already in the format [word, vector]
    import csv as csv
    with open(saveDir, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)

