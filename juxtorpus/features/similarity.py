"""
Similarity between 2 Corpus.

1. jaccard similarity
2. pca similarity
"""
from scipy.spatial import distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

from juxtorpus.corpus import Corpus


class Similarity(object):
    def __init__(self, corpus_A, corpus_B):
        self._A: Corpus = corpus_A
        self._B: Corpus = corpus_B

    def jaccard(self, lemmas: bool = False):
        """ Return a similarity score between the 2 corpus."""
        if lemmas:
            # check if corpus are spacy corpus.
            raise NotImplementedError("To be implemented. Use unique lemmas instead of words.")
        _A_uniqs: set[str] = self._A.unique_words
        _B_uniqs: set[str] = self._B.unique_words
        return len(_A_uniqs.intersection(_B_uniqs)) / len(_A_uniqs.union(_B_uniqs))

    def lsa_pairwise_cosine(self, n_components: int = 100, verbose=False):
        """ Decompose DTM to SVD and return the pairwise cosine similarity of the right singular matrix.

        Note: this may be different to the typical configuration using a TDM instead of DTM.
        However, sklearn only exposes the right singular matrix.
        tdm.T = (U Sigma V.T).T = V.T.T Sigma.T U.T = V Sigma U.T
        the term-topic matrix of U is now the right singular matrix if we use DTM instead of TDM.
        """
        A, B = self._A, self._B
        svd_A = TruncatedSVD(n_components=n_components).fit(A.dtm.tfidf().matrix)
        svd_B = TruncatedSVD(n_components=n_components).fit(B.dtm.tfidf().matrix)
        # TODO: transpose the matrix. It needs to be term vs documents
        if verbose:
            top_terms = 5
            for corpus, svd in [(A, svd_A), (B, svd_B)]:
                feature_indices = svd.components_.argsort()[::-1]
                terms = corpus.dtm.term_names[feature_indices]
                for i in range(feature_indices.shape[0]):
                    print(f"Corpus {str(corpus)}: Singular columns [{i}] {terms[i][:top_terms]}")

        # pairwise cosine
        return cosine_similarity(svd_A.components_, svd_B.components_)

    def tf_cosine(self, without_terms: list[str] = None):
        if without_terms is None:
            return self._tf_cosine(self._A.dtm, self._B.dtm)
        else:
            with self._A.dtm.without_terms(without_terms) as subdtm_A:
                with self._B.dtm.without_terms(without_terms) as subdtm_B:
                    return self._tf_cosine(subdtm_A, subdtm_B)

    def _tf_cosine(self, dtm_0, dtm_1):
        a, b = dtm_0.total_terms_vector, dtm_1.total_terms_vector
        a, b = np.asarray(a).squeeze(axis=0), np.asarray(b).squeeze(axis=0)
        if not a.any() or not b.any():  # zero vectors
            sim, top_terms = 1, None
        else:
            sim = 1 - distance.cosine(a, b)
            highest_similarity_words = np.argsort(a * b)[::-1][:10]  # top 10, element-wise multiplication
            top_terms = self._A.dtm.term_names[highest_similarity_words]
        return {
            'similarity': sim,
            'top_terms': top_terms
        }
        # return np.multiply(a, b).sum() / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()))

    def tfidf_cosine(self):
        """ Compute the cosine similarity between tfidf vectors of 2 corpus.

        The idf should remove the frequent words and manual removal of them e.g. stopwords is not required.
        """
        a = np.asarray(self._A.dtm.tfidf().total_terms_vector).squeeze(axis=0)
        b = np.asarray(self._B.dtm.tfidf().total_terms_vector).squeeze(axis=0)

        highest_similarity_words = np.argsort((a * b))[::-1][:10]  # top 10
        terms = self._A.dtm.term_names[highest_similarity_words]
        return {
            'similarity': 1 - distance.cosine(a, b),
            'top_terms': terms
        }

    def chi_squared_test(self):
        """ Compare the observed and expected frequency by computing the chi squared statistic."""
        root = self._A.find_root()
        if root is not self._B.find_root(): raise ValueError("Comparing corpus must share the same root.")
        expected = np.asarray(root.dtm.total_terms_vector * (self._A.dtm.total / root.dtm.total)).squeeze(axis=0)
        observed = np.asarray(self._A.dtm.total_terms_vector).squeeze(axis=0)
        statistic = self._chi_squared_statistic(observed, expected)
        return {
            'chi_squared': statistic,
            'ddof': len(root.dtm.term_names) - 1,
        }

    def _chi_squared_statistic(self, observed, expected):
        return ((observed - expected) ** 2 / expected).sum()


if __name__ == '__main__':
    import pandas as pd

    corpus = Corpus.from_dataframe(
        pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                    usecols=['processed_text', 'tweet_lga']),
        # pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv',
        #             usecols=['processed_text', 'tweet_lga']),
        col_text='processed_text'
    )
    from juxtorpus.corpus import CorpusSlicer

    slicer = CorpusSlicer(corpus)
    brisbane = slicer.filter_by_item('tweet_lga', 'Brisbane (C)')
    fairfield = slicer.filter_by_item('tweet_lga', 'Fairfield (C)')

    sim = Similarity(brisbane, fairfield)
    pairwise = sim.lsa_pairwise_cosine(n_components=100, verbose=True)
    print(f"SVD pairwise cosine of PCs\n{pairwise}")
    jaccard = sim.jaccard()
    print(f"{jaccard=}")

    sw = nltk.corpus.stopwords.words('english')
    term_vec_cos = sim.tf_cosine(without_terms=sw + ['climate', 'tweet', 'https'])
    print(f"{term_vec_cos=}")

    tfidf_vec_cos = sim.tfidf_cosine()
    print(f"{tfidf_vec_cos=}")
    tfidf_vec_cos = sim.tfidf_cosine()
    print(f"sublinear: {tfidf_vec_cos=}")

    chi_squared = sim.chi_squared_test()
    print(f"{chi_squared=}")
    print()
