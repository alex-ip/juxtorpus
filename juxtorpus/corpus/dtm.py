import contextlib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Iterable, TypeVar

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

TVectorizer = TypeVar('TVectorizer', bound=CountVectorizer)


class DTM(object):

    @classmethod
    def from_wordlists(cls, wordlists: Iterable[Iterable[str]]):
        return cls().build(wordlists)

    def __init__(self):
        self.root = self
        self._vectorizer = None
        self._matrix = None
        self._feature_names_out = None
        self._term_idx_map = None
        self._is_built = False
        self.derived_from = None  # for any dtms derived from word frequencies

        # only used for child dtms
        self._row_indices = None
        self._col_indices = None

    @property
    def is_built(self) -> bool:
        return self.root._is_built

    @property
    def matrix(self):
        matrix = self.root._matrix
        if self._row_indices is not None:
            matrix = matrix[self._row_indices, :]
        if self._col_indices is not None:
            matrix = matrix[:, self._col_indices]
        return matrix

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def num_terms(self):
        return self.matrix.shape[1]

    @property
    def num_docs(self):
        return self.matrix.shape[0]

    @property
    def total(self):
        return self.matrix.sum()

    @property
    def total_terms_vector(self):
        """ Returns a vector of term likelihoods """
        return self.matrix.sum(axis=0)

    @property
    def vectorizer(self):
        return self.root._vectorizer

    @property
    def term_names(self):
        """ Return the terms in the current dtm. """
        features = self.root._feature_names_out
        return features if self._col_indices is None else features[self._col_indices]

    @property
    def vocab(self):
        """ Returns a set of terms in the current dtm. """
        return set(self.term_names)

    def build(self, texts: Iterable[str], vectorizer: TVectorizer = CountVectorizer(token_pattern=r'(?u)b\w+\b')):
        self.root._vectorizer = vectorizer
        self.root._matrix = self.root._vectorizer.fit_transform(texts)
        self.root._feature_names_out = self.root._vectorizer.get_feature_names_out()
        self.root._term_idx_map = {self.root._feature_names_out[idx]: idx
                                   for idx in range(len(self.root._feature_names_out))}
        self.root._is_built = True
        return self

    def terms_column_vectors(self, terms: Union[str, list[str]]):
        """ Return the term vector represented by the documents. """
        cols: Union[int, list[int]]
        if isinstance(terms, str):
            cols = self._term_to_idx(terms)
        else:
            cols = [self._term_to_idx(term) for term in terms]
        return self.matrix[:, cols]

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
        try:
            cloned.matrix
        except Exception as e:
            raise RuntimeError([RuntimeError("Failed to clone DTM."), e])
        return cloned

    def tfidf(self, smooth_idf=True, sublinear_tf=False, norm=None):
        """ Returns an un-normalised tfidf of the current matrix.

        Args: see sklearn.TfidfTransformer
        norm is set to None by default here.
        """
        tfidf_trans = TfidfTransformer(smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, use_idf=True, norm=norm)
        tfidf = DTM()
        tfidf.derived_from = self
        tfidf._vectorizer = tfidf_trans
        tfidf._matrix = tfidf._vectorizer.fit_transform(self.matrix)
        tfidf._feature_names_out = tfidf._vectorizer.get_feature_names_out()
        tfidf._term_idx_map = {tfidf._feature_names_out[idx]: idx for idx in range(len(tfidf._feature_names_out))}
        tfidf._is_built = True
        return tfidf

    def to_dataframe(self):
        return pd.DataFrame.sparse.from_spmatrix(self.matrix, columns=self.term_names)

    @contextlib.contextmanager
    def without_terms(self, terms: Union[list[str], set[str]]):
        """ Expose a temporary dtm object without a list of terms. Terms not found are ignored. """
        try:
            features = self.root._feature_names_out
            self._col_indices = np.isin(features, set(terms), invert=True).nonzero()[0]
            yield self
        finally:
            self._col_indices = None


if __name__ == '__main__':
    from juxtorpus.corpus.corpus import Corpus

    df = pd.read_csv(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
    corpus = Corpus.from_dataframe(df, col_text='processed_text')

    dtm = DTM().build(corpus.texts())
    print(dtm.terms_column_vectors('the').shape)
    print(dtm.terms_column_vectors(['the', 'he', 'she']).shape)

    sub_df = df[df['processed_text'].str.contains('the')]

    child_dtm = dtm.cloned(dtm, sub_df.index)
    print(child_dtm.terms_column_vectors('the').shape)
    print(child_dtm.terms_column_vectors(['the', 'he', 'she']).shape)

    df = child_dtm.to_dataframe()
    print(df.head())

    print(f"Child DTM shape: {child_dtm.shape}")
    print(f"Child DTM DF shape: {df.shape}")
    print(f"Child DTM DF memory usage:")
    df.info(memory_usage='deep')

    # with remove_words context
    prev = set(dtm.term_names)
    with dtm.without_terms({'hello'}) as subdtm:
        print(subdtm.num_terms)
        print(prev.difference(set(subdtm.term_names)))
