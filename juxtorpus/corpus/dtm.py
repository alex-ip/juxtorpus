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
    def __init__(self, lists_of_wordlist: Iterable[Iterable[str]], root_dtm=None):
        self.vectorizer = CountVectorizer()
        self.dtm = self.vectorizer.fit_transform((' '.join(words) for words in lists_of_wordlist))
        self.vocab = self.vectorizer.get_feature_names_out()
        self._term_idx_map = {self.vocab[idx]: idx for idx in range(len(self.vocab))}

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
        if term not in self._term_idx_map.keys(): raise ValueError(f"'{term}' not found in document-term-matrix.")
        return self._term_idx_map.get(term)

    # clone by document
    def cloned(self, doc_indices: list[int]) -> scipy.sparse.csr_matrix:
        return self.dtm[doc_indices]  # TODO: this should 'probably' return a copy of the DTM object itself.


if __name__ == '__main__':
    df = pd.read_csv(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
    corpus = Corpus.from_dataframe(df, col_text='processed_text')

    dtm = DTM(corpus.generate_words())
    print(dtm.term_vector('the').shape)
    print(dtm.term_vector(['the', 'he', 'she']).shape)

    sub_dtm = dtm.cloned([1, 6, 8])
    print(sub_dtm.shape)
    assert pairwise_distances(dtm.dtm[1], sub_dtm[0], metric='cosine')[0][0] == 0.0

# put the DTM in the corpus.
# TODO: try multiplying with IDF -> TFIDFTransformer
