import sys

import gensim


def dummy_metric(truth, hypothesis):
    """
    example of how we can judge whether the answer is correct
    :param truth:
    :param hypothesis:
    :return:
    """
    return hypothesis <= truth + 1 and hypothesis >= truth - 1


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
    # have the option to choose a dimension for glove
    dimension_opt = [50, 100, 200, 300]
    try:
        word_dim = int(sys.argv[1])
    except IndexError:
        word_dim = 50

    try:
        evaluation_file = sys.argv[2]
    except IndexError:
        evaluation_file = '../data/wordsim353.txt'

    assert word_dim in dimension_opt
    # this is the analogy model that we will be using

    embFileName = 'glove.6B.{0}d.txt'.format(word_dim)
    embeddingsFile = '../data/glove.6B/' + embFileName
    outFile = '../results/accuracy_' + embFileName
    gsm_mod = gensim.models.Word2Vec.load_word2vec_format(embeddingsFile, binary=False)
    gsm_mod.init_sims(replace=True)  # indicates we're finished training to save ram
    evaluate_similarity(evaluation_file, gsm_mod)

