import gensim
import nltk
from collections import OrderedDict
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

RESOURCES = ""
RESOURCES = '/home/classes/cs477/data/'

# Given a filename where each line is in the format "<word1>  <word2>  <human_score>", 
# return a dictionary of {(word1, word2):human_score), ...}
# Note that human_scores in your dictionary should be floats.
def parseFile(filename):
    similarities = OrderedDict()
    f = open(filename)
    for line in f:
        token_lst = nltk.word_tokenize(line)
        word_tup = (token_lst[0], token_lst[1])
        score = float(token_lst[2])
        similarities[word_tup] = score
    return similarities
    
# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Lin similarity scores {(word1, word2):similarity_score, ...}
def linSimilarities(word_pairs, ic):
    similarities = {}
    for pair in word_pairs:
        word1 = pair[0]
        word2 = pair[1]
        word1_synsets = wn.synsets(word1, pos=wn.NOUN)
        word2_synsets = wn.synsets(word2, pos=wn.NOUN)
        if not (word1_synsets and word2_synsets):
            word1_synsets = wn.synsets(word1, pos=wn.VERB)
            word2_synsets = wn.synsets(word2, pos=wn.VERB)
        synset1 = word1_synsets[0]
        synset2 = word2_synsets[0]
        word_tup = (word1, word2)
        lin_sim = synset1.lin_similarity(synset2, ic) * 10
        similarities[word_tup] = lin_sim 
    return similarities

# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Resnik  similarity scores {(word1, word2):similarity_score, ...}
def resSimilarities(word_pairs, ic):
    similarities = {}
    for pair in word_pairs:
        word1 = pair[0]
        word2 = pair[1]
        word1_synsets = wn.synsets(word1, pos=wn.NOUN)
        word2_synsets = wn.synsets(word2, pos=wn.NOUN)
        if not (word1_synsets and word2_synsets):
            word1_synsets = wn.synsets(word1, pos=wn.VERB)
            word2_synsets = wn.synsets(word2, pos=wn.VERB)
        synset1 = word1_synsets[0]
        synset2 = word2_synsets[0]
        word_tup = (word1, word2)
        res_sim = synset1.res_similarity(synset2, ic)
        similarities[word_tup] = res_sim 
    return similarities

# Given a list of tuples [(word1, word2), ...] and a word2vec model, return a dictionary 
# of word2vec similarity scores {(word1, word2):similarity_score, ...}
def vecSimilarities(word_pairs, model):
    similarities = {}
    for pair in word_pairs:
        word1 = pair[0]
        word2 = pair[1]
        word_tup = (word1, word2)
        similarity = model.similarity(word1.lower(), word2.lower()) * 10
        similarities[word_tup] = similarity
    return similarities


def main():
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    human_sims = parseFile("input.txt")

    lin_sims = linSimilarities(human_sims.keys(), brown_ic)
    res_sims = resSimilarities(human_sims.keys(), brown_ic)

    model = None
    model = gensim.models.Word2Vec()
    model = model.load_word2vec_format(RESOURCES+'glove_model.txt', binary=False)
    vec_sims = vecSimilarities(human_sims.keys(), model)
    
    lin_score = 0
    res_score = 0
    vec_score = 0

    print '{0:15} {1:15} {2:10} {3:20} {4:20} {5:20}'.format('word1','word2', 
                                                             'human', 'Lin', 
                                                             'Resnik', 'Word2Vec')
    for key, human in human_sims.items():
        try:
            lin = lin_sims[key]
        except:
            lin = 0
        lin_score += (lin - human) ** 2
        try:
            res = res_sims[key]
        except:
            res = 0
        res_score += (res - human) ** 2
        try:
            vec = vec_sims[key]
        except:
            vec = 0
        vec_score += (vec - human) ** 2
        print '{0:15} {1:15} {2:10} {3:20} {4:20} {5:20}'.format(key[0], key[1], human, 
                                                                 lin, res, vec)

    num_examples = len(human_sims)
    print "\nMean Squared Errors"
    print "Lin method error: %0.2f" % (lin_score/num_examples) 
    print "Resnick method error: %0.2f" % (res_score/num_examples)
    print "Vector-based method error: %0.2f" % (vec_score/num_examples)

if __name__ == "__main__":
    main()
