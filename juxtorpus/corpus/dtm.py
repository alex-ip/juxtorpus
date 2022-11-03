""" Document Term Matrix

feature:
1. always build dtm for corpus
2. lazy dtm

test: cloning works
"""

from pathlib import Path
import pandas as pd
import scipy.sparse
from typing import Iterable, Union, Generator

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

from juxtorpus.corpus import Corpus

""" Document Term Matrix DTM

DTM is a container for the document term sparse matrix.
It is a component of the corpus class.
"""


class DTM(object):

    @classmethod
    def from_wordlists(cls, wordlists: list[list[str]]):
        return cls().build(wordlists)

    def __init__(self):
        self.root = self
        self.vectorizer = CountVectorizer()
        self._dtm = None
        self._vocab = None
        self._term_idx_map = None

        # only used for child dtms
        self._row_indices = None

    @property
    def dtm(self):
        if self.root is self:
            return self._dtm
        return self.root._dtm[self._row_indices, :]

    @property
    def vocab(self):
        return list(self.root._vocab)

    def build(self, wordlists: list[list[str]]):
        self._dtm = self.vectorizer.fit_transform((' '.join(words) for words in wordlists))
        self._vocab = self.vectorizer.get_feature_names_out()
        self._term_idx_map = {self.vocab[idx]: idx for idx in range(len(self._vocab))}
        return self

    def term_vector(self, terms: Union[str, list[str]]):
        """ Return the term vector represented by the documents. """
        cols: Union[int, list[int]]
        if isinstance(terms, str):
            cols = self._term_to_idx(terms)
        else:
            cols = [self._term_to_idx(term) for term in terms]
        return self.dtm[:, cols]

    def doc_vector(self):  # TODO: from pandas index?
        """ Return the document vector represented by the terms. """
        raise NotImplementedError()

    def _term_to_idx(self, term: str):
        if term not in self.root._term_idx_map.keys(): raise ValueError(f"'{term}' not found in document-term-matrix.")
        return self.root._term_idx_map.get(term)

    @classmethod
    def cloned(cls, parent, row_indices: list[int]):
        cloned = cls()
        cloned.root = parent.root
        cloned._row_indices = row_indices
        return cloned


if __name__ == '__main__':
    df = pd.read_csv(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
    corpus = Corpus.from_dataframe(df, col_text='processed_text')

    dtm = DTM.from_wordlists(corpus.generate_words())
    print(dtm.term_vector('the').shape)
    print(dtm.term_vector(['the', 'he', 'she']).shape)

    child_dtm = dtm.cloned(dtm, [1, 5, 7, 8])
    print(child_dtm.term_vector('the').shape)
    print(child_dtm.term_vector(['the', 'he', 'she']).shape)

    # PERF: child dtms slices root dtm each time -- seems negligible since its array access anyway.
    from timeit import timeit

    start = timeit()
    for i in range(1000000):
        dtm.term_vector('the')
    print(f"Elapsed: {timeit() - start:.100f} s")
    start = timeit()
    for i in range(1000000):
        child_dtm.term_vector('the')
    print(f"Elapsed: {timeit() - start:.100f} s")
