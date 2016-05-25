#this code writes data in different formats (csv, space_separated, etc.)

def writeCSV(words,matrix, saveDir):
    import csv
    #words: is a numpy array or an array
    #matrix: must be a NUMERICAL NUMPY ARRAY
    #first build the "joint" matrix
    MATRIX = []
    for i in range(len(words)):
        vec = matrix[i,]  # WARNING: for now we just pick the first one!
        currentWord = [ words[i] ]
        currentWord.extend(vec)
        MATRIX.append(currentWord)
    # write the CSV file
    with open(saveDir, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)
    return MATRIX


#TODO: write to space-separated file
