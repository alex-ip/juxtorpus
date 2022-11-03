from pathlib import Path
import pandas as pd
from typing import Union

from sklearn.feature_extraction.text import CountVectorizer

from juxtorpus.corpus import Corpus

""" Document Term Matrix DTM

DTM is a container for the document term sparse matrix.
This container allows you to access term vectors and document vectors.
It also allows you to clone itself with a row index. In the cloned DTM,
a reference to the root dtm is passed down and the row index is used to
slice the child dtm each time.
This serves 3 purposes:
    1. preserve the original/root vocabulary.
    2. performance reasons so that we don't need to rebuild a dtm each time.
    3. indexing is a very inexpensive operation.

Dependencies: 
sklearn CountVectorizer
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
    def cloned(cls, parent, row_indices: Union[pd.core.indexes.numeric.Int64Index, list[int]]):
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

    sub_df = df[df['processed_text'].str.contains('the')]

    child_dtm = dtm.cloned(dtm, sub_df.index)
    print(child_dtm.term_vector('the').shape)
    print(child_dtm.term_vector(['the', 'he', 'she']).shape)
