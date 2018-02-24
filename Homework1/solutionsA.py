import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    unigram_count = 0.0

    for sentence in training_corpus:
        # Tokenize input
        tokens = sentence.split()
        tokens.append(STOP_SYMBOL)

        #Unigrams
        for word in tokens:
            unigram_count += 1.0
            unigram = (word,)
            if unigram in unigram_p:
                unigram_p[unigram] += 1.0
            else:
                unigram_p[unigram] = 1.0

        #Bigrams
        tokens.insert(0, START_SYMBOL)
        bigram_t = list(nltk.bigrams(tokens))
        for bigram in bigram_t:
            if bigram in bigram_p:
                bigram_p[bigram] += 1.0
            else:
                bigram_p[bigram] = 1.0

        #Trigrams
        tokens.insert(0, START_SYMBOL)
        trigram_t = list(nltk.trigrams(tokens))
        for trigram in trigram_t:
            if trigram in trigram_p:
                trigram_p[trigram] += 1.0
            else:
                trigram_p[trigram] = 1.0

    # Trigram Probabilities
    for trigram in trigram_p:
        if trigram[0] == '*' and trigram[1] == '*':
            trigram_p[trigram] = math.log(trigram_p[trigram]/unigram_p[(STOP_SYMBOL,)], 2)
        else:
            trigram_p[trigram] = math.log(trigram_p[trigram]/bigram_p[(trigram[0], trigram[1])], 2)

    # Bigram Probabilities
    for bigram in bigram_p:
        if bigram[0] == '*':
            bigram_p[bigram] = math.log(bigram_p[bigram]/unigram_p[(STOP_SYMBOL,)], 2)
        else:
            bigram_p[bigram] = math.log(bigram_p[bigram]/unigram_p[(bigram[0],)], 2)

    #Unigram Probabilities
    for unigram in unigram_p:
        unigram_p[unigram] = math.log(unigram_p[unigram]/unigram_count, 2)
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for sentence in corpus:
        token_score = []
        tokens = sentence.split()
        tokens.append(STOP_SYMBOL)
        # Unigram
        if n == 1:
            for word in tokens:
                word = (word,)
                if word in ngram_p:
                    token_score.append(ngram_p[word])
                else:
                    token_score[:] = []
                    token_score.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
            scores.append(sum(token_score))
        # Bigram
        elif n == 2:
            tokens.insert(0, START_SYMBOL)
            bigram_t = nltk.bigrams(tokens)
            for bigram in bigram_t:
                if bigram in ngram_p:
                    token_score.append(ngram_p[bigram])
                else:
                    token_score[:] = []
                    token_score.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
            scores.append(sum(token_score))
        # Trigram
        elif n == 3:
            tokens.insert(0, START_SYMBOL)
            tokens.insert(0, START_SYMBOL)
            trigram_t = nltk.trigrams(tokens)
            for trigram in trigram_t:
                if trigram in ngram_p:
                    token_score.append(ngram_p[trigram])
                else:
                    token_score[:] = []
                    token_score.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
            scores.append(sum(token_score))

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for sentence in corpus:
        tokens = sentence.split()
        tokens.insert(0, START_SYMBOL)
        tokens.insert(0, START_SYMBOL)
        tokens.append(STOP_SYMBOL)
        sentence_score = []

        trigram_t = nltk.trigrams(tokens)
        for trigram in trigram_t:
            uni_p, bi_p, tri_p = 0, 0, 0

            # Calculate unigram probability
            unigram_key = (trigram[2],)
            if unigram_key in unigrams:
                uni_p = 2.0**unigrams[unigram_key]

            # Calculate bigram probability
            bigram_key = (trigram[1], trigram[2])
            if bigram_key in bigrams:
                bi_p = 2.0**bigrams[bigram_key]

            # Calculate trigram probability
            if trigram in trigrams:
                tri_p = 2.0**trigrams[trigram]

            if uni_p == 0 and bi_p == 0 and tri_p == 0:
                sentence_score[:] = []
                sentence_score.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                break
            else:
                sentence_score.append(math.log((uni_p + bi_p + tri_p)/3.0, 2))
        scores.append(sum(sentence_score))
    return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
