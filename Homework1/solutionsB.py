import sys
from collections import Counter
from itertools import chain
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the words of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        tok_words = []
        tok_tags = []
        tokens = sentence.split()
        tokens.insert(0, START_SYMBOL)
        tokens.insert(0, START_SYMBOL)
        tokens.append(STOP_SYMBOL)

        for token in tokens:
            token_split = token.rsplit('/', 1)
            word = token_split[0]
            if len(token_split) == 2:
                tag = token_split[1]
            else:
                tag = token_split[0]
            tok_tags.append(tag)
            tok_words.append(word)
        brown_words.append(tok_words)
        brown_tags.append(tok_tags)
    return brown_words, brown_tags

# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    unigram_p = {}
    bigram_p = {}

    for sentence in brown_tags: 
        # Unigrams
        unigram_t = sentence[2:]
        for unigram in unigram_t:
            unigram = (unigram,)
            if unigram in unigram_p:
                unigram_p[unigram] += 1.0
            else:
                unigram_p[unigram] = 1.0

        # Bigrams
        bigram_t = list(nltk.bigrams(sentence[1:]))
        for bigram in bigram_t:
            if bigram in bigram_p:
                bigram_p[bigram] += 1.0
            else:
                bigram_p[bigram] = 1.0

        # Trigrams
        trigram_t = list(nltk.trigrams(sentence))
        for trigram in trigram_t:
            if trigram in q_values:
                q_values[trigram] += 1.0
            else:
                q_values[trigram] = 1.0

    # Trigram Probability
    for trigram in q_values:
        if trigram[0] == '*' and trigram[1] == '*':
            q_values[trigram] = math.log(q_values[trigram]/unigram_p[(STOP_SYMBOL,)], 2)
        else:
            q_values[trigram] = math.log(
                    q_values[trigram]/bigram_p[(trigram[0], trigram[1])], 2)

    return q_values

#TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates the tag trigrams in reverse.  In other words, instead of looking at the probabilities that the third tag follows the first two, look at the probabilities of the first tag given the next two.
# Hint: This code should only differ slightly from calc_trigrams(brown_tags)
def calc_trigrams_reverse(brown_tags):
    q_values = {}
    
    unigram_p = {}
    bigram_p = {}

    for sentence in brown_tags: 
        # Unigrams
        unigram_t = sentence[2:]
        for unigram in unigram_t:
            unigram = (unigram,)
            if unigram in unigram_p:
                unigram_p[unigram] += 1.0
            else:
                unigram_p[unigram] = 1.0

        # Bigrams
        bigram_t = list(nltk.bigrams(sentence[1:]))
        for bigram in bigram_t:
            if bigram in bigram_p:
                bigram_p[bigram] += 1.0
            else:
                bigram_p[bigram] = 1.0

        # Trigrams
        trigram_t = list(nltk.trigrams(sentence))
        for trigram in trigram_t:
            new_t = (trigram[2], trigram[1], trigram[0])
            if new_t in q_values:
                q_values[new_t] += 1.0
            else:
                q_values[new_t] = 1.0

    # Trigram Probability
    for trigram in q_values:
        if trigram[0] == '*' and trigram[1] == '*':
            q_values[trigram] = math.log(q_values[trigram]/unigram_p[(STOP_SYMBOL,)], 2)
        else:
            q_values[trigram] = math.log(
                    q_values[trigram]/bigram_p[(trigram[1], trigram[0])], 2)

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])

    all_words = list(chain.from_iterable(brown_words))
    wordcount = Counter(all_words)

    for key, value in wordcount.items():
        if value > RARE_WORD_MAX_FREQ:
            known_words.add(key)

    return known_words

# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = brown_words 
    for sentence in brown_words:
        for i, word in enumerate(sentence):
            if word not in known_words:
                sentence[i] = RARE_SYMBOL
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])

    # Flatten
    brown_words_rare = list(chain.from_iterable(brown_words_rare))
    brown_tags = list(chain.from_iterable(brown_tags))
    tags_count = Counter(brown_tags)

    # Counts
    for pair in zip(brown_words_rare, brown_tags):
        if pair in e_values:
            e_values[pair] += 1.0
        else:
            e_values[pair] = 1.0
        tag = pair[1]
        taglist.add(tag)

    # Calculate emission probability
    for pair in e_values:
        e_values[pair] = math.log(e_values[pair]/tags_count[pair[1]], 2)
    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def return_taglist(k, taglist):
    if k >= 1:
        return taglist - set([START_SYMBOL, STOP_SYMBOL])
    else:
        return set('*')

def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    brown_copy = list(brown_dev_words)
    for item in brown_copy:
        # Initialize
        sentence = list(item)
        bp = {}
        pi = {}
        pi[(-1, START_SYMBOL, START_SYMBOL)] = 0
        pi[(0, START_SYMBOL, START_SYMBOL)] = 0 
        n = len(sentence)
        sentence.insert(0, 'dummy_elem')

        # Viterbifor k in range(2, sen_len-1):
        for k in range(1, n+1):
            word = sentence[k]
            if word not in known_words:
                word = RARE_SYMBOL
            u_tags = return_taglist(k-1, taglist)
            for u in u_tags:
                v_tags = return_taglist(k, taglist)
                for v in v_tags:
                    max_p = 1000*LOG_PROB_OF_ZERO
                    max_arg = v 
                    pair = (word, v)

                    if pair in e_values:
                        w_tags = return_taglist(k-2, taglist)
                        for w in w_tags:
                            trigram = (w, u, v)
                            if trigram not in q_values:
                                q_values[trigram] = LOG_PROB_OF_ZERO
                            prob = pi[(k-1, w, u)] + q_values[trigram] + e_values[pair]
                            if prob > max_p:
                                max_p = prob
                                max_arg = w
                    pi[(k, u, v)] = max_p
                    bp[(k, u, v)] = max_arg

        # STOP end case
        max_u = None
        max_v = None
        max_p = 1000*LOG_PROB_OF_ZERO
        u_tags = return_taglist(n-1, taglist)
        for u in u_tags:
            v_tags = return_taglist(n, taglist)
            for v in v_tags:
                trigram = (u, v, STOP_SYMBOL)
                if trigram not in q_values:
                    q_values[trigram] = LOG_PROB_OF_ZERO
                prob = q_values[trigram] + pi[(n, u, v)]
                if prob > max_p:
                    max_u = u
                    max_v = v
        
        # List of tags
        tags = [''] * (n+1)
        tags[n] = max_v
        tags[n-1] = max_u

        # Back pointer
        #tags = [bp[(k+2, tags[k+1], tags[k+2])] for k in range(n-2, 0, -1)]
        for k in range(n-2, 0, -1):
            tags[k] = bp[(k+2, tags[k+1], tags[k+2])]
        
        # For zero indexing, resize tags list
        tags = tags[1:]
        sentence = sentence[1:]

        # Attach tags
        tagged.append(' '.join([sentence[k] + '/' + tags[k] for k in range(n)]) + '\n')
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    
    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for sentence in brown_dev_words:
        tagged_sentence = trigram_tagger.tag(sentence)
        words = '' 
        for tag in tagged_sentence:
            words += tag[0] + '/' + tag[1] + ' '
        words += '\n'
        tagged.append(words)
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities with an option to reverse (question 7)
    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = calc_trigrams_reverse(brown_tags)
    else:
        q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    #nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    #q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
