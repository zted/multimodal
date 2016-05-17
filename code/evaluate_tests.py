"""
Evaluates word embeddings against a similarity test set

Instructions to run: cd into topmost directory
type "python code/evaluate_tests.py embeddings/my_embeddings.txt [optional_test_file.txt]
note that my_embeddings.txt needs to contain embeddings corresponding
to all the vocabulary in the test file
"""
import sys

import gensim


def dummy_metric(truth, hypothesis):
    """
    example of how we can judge whether the answer is correct
    :param truth:
    :param hypothesis:
    :return:
    """
    return truth - 1 <= hypothesis <= truth + 1


def dummy_score_eval(correct, incorrect):
    """
    example of how we can judge the final score
    :param correct:
    :param incorrect:
    :return:
    """
    return correct / float(correct + incorrect)


def load_test_file(tf):
    word_pairs = []
    scores = []
    with open(tf, 'r') as f:
        for line in f:
            splits = line.rstrip('\n').split('\t')
            word_pairs.append((splits[0], splits[1]))
            scores.append(float(splits[2]))
    return word_pairs, scores


def evaluate_similarity(eval_file, gsm_mod):
    correct = 0
    incorrect = 0
    wordpairs, scores = load_test_file(eval_file)
    for i in range(len(wordpairs)):
        w1 = wordpairs[i][0].lower()
        w2 = wordpairs[i][1].lower()
        score = scores[i]
        guess = gsm_mod.similarity(w1, w2) * 10
        if dummy_metric(score, guess):
            correct += 1
        else:
            incorrect += 1

    score = dummy_score_eval(correct, incorrect)
    print("Score: {}".format(score))
    return


if __name__ == "__main__":

    try:
        EMBEDDINGSFILE = sys.argv[1]
    except IndexError as e:
        print('Must enter embeddings file to use!')
        raise e

    try:
        evaluation_file = sys.argv[2]
    except IndexError:
        evaluation_file = 'data/simlex999.txt'

    gsm_mod = gensim.models.Word2Vec.load_word2vec_format(EMBEDDINGSFILE, binary=False)
    gsm_mod.init_sims(replace=True)  # indicates we're finished training to save ram
    evaluate_similarity(evaluation_file, gsm_mod)
