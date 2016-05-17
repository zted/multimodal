"""
Given a file with visual embeddings and file with word embeddings,
output a file with multimodal embeddings.
"""
import gensim
import sys
import numpy as np


def dummy_map(word_model, visual_model):
    """
    replace with svm, nn, whatever
    :param
    :return:
    """
    output = []
    for vocab in word_model.vocab:
        word_vec = word_model[vocab]
        vis_vec = word_vec + 1
        # replace the previous word embedding with concatenated one
        output.append((vocab, np.concatenate((word_vec, vis_vec))))
    return word_model


def zero_padding_map(word_model, visual_model):
    """
    simplest mapping model, concatenate those words
    with visual vectors, zero pad those without and
    return the new vectors
    :param
    :return: tuple formatted as: (word, numpy array)
    """
    output = []
    for vocab in word_model.vocab:
        word_vec = word_model[vocab]
        vocab_dim = len(word_vec)
        try:
            vis_vec = visual_model[vocab]
        except KeyError:
            # word cannot be found, we zero pad it
            vis_vec = np.zeros(vocab_dim)
        # replace the previous word embedding with concatenated one
        output.append((vocab, np.concatenate((word_vec, vis_vec))))
    return output


def savew2v_format(tuple_list, somefile):
    vocab_dim = len(tuple_list[0][1])
    f = open(somefile, 'w')
    f.write('{} {}\n'.format(len(tuple_list), vocab_dim))
    for w, v in tuple_list:
        v = ' '.join(map(str, v))
        f.write('{} {}\n'.format(w.encode('utf-8'), v))
    f.close()
    return


def create_mm_embeddings(visual_embeddings, word_embeddings, outputfile, mapping_function=zero_padding_map):
    word_mod = gensim.models.Word2Vec.load_word2vec_format(word_embeddings, binary=False)
    vis_mod = gensim.models.Word2Vec.load_word2vec_format(visual_embeddings, binary=False)
    word_mod = mapping_function(word_mod, vis_mod)
    savew2v_format(word_mod, outputfile)
    return


if __name__ == "__main__":

    OUTPUTFILE = 'embeddings/combined_embeddings.txt'
    try:
        EMBEDDINGSFILE = sys.argv[1]
    except IndexError as e:
        print('Must enter word embeddings file to use!')
        raise e

    try:
        VISUALEMBEDDINGS = sys.argv[2]
    except IndexError as e:
        print('Must enter visual embeddings file to use!')
        raise e

    try:
        OUTPUTFILE = sys.argv[3]
    except IndexError as e:
        pass

    create_mm_embeddings(VISUALEMBEDDINGS, EMBEDDINGSFILE, OUTPUTFILE)
