# Use Cases and Methodology

This document describes in the form of a process the possible use cases in comparing
the linguistic phenomena of two corpora.

### Pre: Some Assumptions of the user

The user is assumed to have chosen 2 corpus that are only different in 1 aspect (taking the ideal case here maybe?)
when *juxtorpusing* them to reveal linguistic distinctions within this particular context defined by the aspect.

A concrete example:
Comparing two corpora of tweets from two different authors but in the same time interval (during 2019-2020)

### 1. Linguistic makeup of corpus (characteristic phrases)

A: Theme identification

1. Extract the keywords from the corpora
2. Compare their frequencies
3. Study the collocations, using concordance, of these keywords
    1. perhaps identify any shifts across time.
        1. using visualisations OR frequencies
    2. perhaps identify patterns behind why they're keywords.
4. (Out of scope) whether this shift is based on chance or not.

B: Assuming Subject-Verb-Object

1. Extract the frequently used nouns (could be the subject or object)
2. Show them in their noun phrases
3. Show them with associated verbs
4. Use word embeddings to cluster those words with kmeans (with cosine to assuage varying document lengths).
   If sentences includes the words such as 'and' (CC - coordinating conjunction), then we break the sentences up.
5. note: word embeddings will not work well with entities.

### 2. Entities mentioned in the corpus

1. List the most frequently mentioned entities in the corpus.

### 2. Topic modelling of corpus

* SVD/LSA
* LDA

### 3. Clustering - (requires input from Sony)

* KMeans
* DBScan