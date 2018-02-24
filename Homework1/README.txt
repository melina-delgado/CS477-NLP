Melina Delgado
5429137808

Part A1
UNIGRAM natural -13.766408817
BIGRAM natural that -4.05889368905
TRIGRAM natural that he -1.58496250072

Part A2
python perplexity.py output/A2.uni.txt data/Brown_train.txt 
The perplexity is 1052.4865859

python perplexity.py output/A2.bi.txt data/Brown_train.txt 
The perplexity is 53.8984761198

python perplexity.py output/A2.tri.txt data/Brown_train.txt
The perplexity is 5.7106793082

Part A3
python perplexity.py output/A3.txt data/Brown_train.txt 
The perplexity is 12.5516094886

Part A4
For the models without interpolation, it makes sense that trigram returns the lowest
perplexity in comparison to bigrams and unigrams. The more information the N-
gram gives us about the word sequence, the lower the perplexity.
The perplexity returned by the linear interpolation is a bit higher because we are 
relying on trigram, unigram and bigram probabilities. 

Part A5
python perplexity.py output/Sample1_scored.txt data/Sample1.txt
The perplexity is 11.1670289158
python perplexity.py output/Sample2_scored.txt data/Sample2.txt
The perplexity is 1611240282.44

Given these two perplexities, I can infer that data/Sample1.txt belongs to the Brown
training dataset. This is because sometimes it is possible to achieve lower perplexity
on a more predictable corpora. Because my program used the Brown dataset for training,
it makes sense that it will perform better on perplexity for a sample of the Brown dataset.
A low perplexity indicates the probability distribution is good at predicting the sample.
Therefore, this means my model was better at prediction Sample1, probably because it 
belongs to the Brown dataset. The words in Sample2 are unseen, so my model set all 
sentence probabilities to -1000.

Part B2
TRIGRAM CONJ ADV ADP -2.9755173148
TRIGRAM DET NOUN NUM -8.9700526163
TRIGRAM NOUN PRT PRON -11.0854724592

Part B4
* * 0.0
Night NOUN -13.8819025994
Place VERB -15.4538814891
prime ADJ -10.6948327183
STOP STOP 0.0
_RARE_ VERB -3.17732085089

Part B5
python pos.py output/B5.txt data/Brown_tagged_dev.txt
Percent correct tags: 91.8134735279

Part B6
python pos.py output/B6.txt data/Brown_tagged_dev.txt
Percent correct tags: 86.7918775773

Part B7
python pos.py output/B5.txt data/Brown_tagged_dev.txt
Percent correct tags: 82.8963980691

Our normal tagger achieves a greater accuracy than the reverse tagger because the 
normal tagger has the correct interpretation of context. The reverse tagger will
tag differently because reversing a sentence will change its context.

A reverse-trigram-trained POS tagger could work better than a normally trained
POS tagger if the likelihood of the reverse sentence is greater than the original
one. Probably some things Yoda says fall under this. Sayings like "right you are"
become "are you right" (if we ignore some punctuation) and other inverted sentences
might produce a higher probability in a reverse trigram.

Part C
The Spanish dataset takes longer to evaluate because it may be larger. Also, because
we didn't train on that data, that is also probably why.

Contexts in different languages, orders of parts of speech, new parts of speech are
things that are not captured by training sets in a different language.

Final Run:
A
python solutionsA.py
Part A time: 8.968311 sec

B
python solutionsB.py
Part B time: 95.218783 sec

B reverse
python solutionsB.py -reverse
Part B time: 95.698847 sec

C
C reverse
