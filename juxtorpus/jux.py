from juxtorpus.corpus import Corpus
from juxtorpus.stats import Statistics
from juxtorpus.features.similarity import Similarity
from juxtorpus.features.keywords import Keywords, RakeKeywords, TFKeywords, TFIDFKeywords

import numpy as np
from typing import TypeVar

CorpusT = TypeVar('CorpusT', bound=Corpus)  # Corpus subclass


class Jux:
    """ Jux
    This is the main class for Juxtorpus. It takes in 2 corpus and exposes numerous functions
    to help contrast the two corpus.

    It is expected that the exposed functions are used as tools that serve as building blocks
    for your own further analysis.
    """

    def __init__(self, corpus_a: CorpusT, corpus_b: CorpusT):
        self._A = corpus_a
        self._B = corpus_b
        self._stats = Statistics(self)
        self._sim = Similarity(self._A, self._B)

    @property
    def stats(self):
        return self._stats

    @property
    def sim(self):
        return self._sim

    @property
    def num_corpus(self):
        return 2

    @property
    def corpus_a(self):
        return self._A

    @property
    def corpus_b(self):
        return self._B

    @property
    def shares_parent(self) -> bool:
        return self._A.find_root() is self._B.find_root()

    def keywords(self, method: str):
        """ Extract and return the keywords of the two corpus ranked by frequency. """
        extractor_A: Keywords
        extractor_B: Keywords
        if method == 'rake':
            extractor_A = RakeKeywords(corpus=self._A)
            extractor_B = RakeKeywords(corpus=self._B)
        elif method == 'tf':
            extractor_A = TFKeywords(corpus=self._A)
            extractor_B = TFKeywords(corpus=self._B)
        elif method == 'tfidf':
            extractor_A = TFIDFKeywords(corpus=self._A)
            extractor_B = TFIDFKeywords(corpus=self._B)
        else:
            raise ValueError("Unsupported keyword extraction method.")
        # use sets to compare
        return extractor_A.extracted(), extractor_B.extracted()

    def lexical_diversity(self):
        """ Return the lexical diversity comparison"""
        # a smaller corpus will generally have better lexical diversity
        ld_A = self._A.num_unique_words / np.log(self._A.num_words)
        ld_B = self._B.num_unique_words / np.log(self._B.num_words)
        return ld_A - ld_B, {'corpusA': ld_A, 'corpusB': ld_B}
