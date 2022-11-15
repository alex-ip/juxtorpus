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

    def svd_pairwise_cosine(self, n_components: int, verbose=False):
        """ Eigen decompose DTM and return the pairwise cosine similarity of the principal components. """
        svd_A = TruncatedSVD(n_components=n_components).fit(self._A.dtm.matrix)
        svd_B = TruncatedSVD(n_components=n_components).fit(self._B.dtm.matrix)

        if verbose:
            top_terms = 5
            for corpus, svd in [(self._A, svd_A), (self._B, svd_B)]:
                feature_indices = svd.components_.argsort()[::-1]
                terms = corpus.dtm.term_names[feature_indices]
                for i in range(feature_indices.shape[0]):
                    print(f"Corpus {str(corpus)}: PC [{i}] {terms[i][:top_terms]}")

        # pairwise cosine
        return cosine_similarity(svd_A.components_, svd_B.components_)

    def tf_cosine(self, without_terms: list[str] = None):
        if without_terms is None: without_terms = list()
        with self._A.dtm.without_terms(without_terms) as subdtm_A:
            with self._B.dtm.without_terms(without_terms) as subdtm_B:
                a, b = subdtm_A.total_terms_vector, subdtm_B.total_terms_vector
                a, b = np.asarray(a).squeeze(axis=0), np.asarray(b).squeeze(axis=0)
                # return np.multiply(a, b).sum() / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()))
                highest_similarity_words = np.argsort((a * b))[::-1][:10]  # top 10
                terms = self._A.dtm.term_names[highest_similarity_words]
                return {
                    'similarity': 1 - distance.cosine(a, b),
                    'top_terms': terms
                }

    def tfidf_cosine(self):
        """ Compute the cosine similarity between tfidf vectors of 2 corpus.

        The idf should remove the frequent words and manual removal of them e.g. stopwords is not required.
        """
        a = np.asarray(self._A.dtm.total_terms_vector).squeeze(axis=0)
        b = np.asarray(self._B.dtm.total_terms_vector).squeeze(axis=0)

        highest_similarity_words = np.argsort((a * b))[::-1][:10]  # top 10
        terms = self._A.dtm.term_names[highest_similarity_words]
        return {
            'similarity': 1 - distance.cosine(a, b),
            'top_terms': terms
        }


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
    pairwise = sim.svd_pairwise_cosine(n_components=3, verbose=True)
    print(f"SVD pairwise cosine of PCs\n{pairwise}")
    jaccard = sim.jaccard()
    print(f"{jaccard=}")

    sw = nltk.corpus.stopwords.words('english')
    term_vec_cos = sim.tf_cosine(without_terms=sw + ['climate', 'tweet', 'https'])
    print(f"{term_vec_cos=}")

    tfidf_vec_cos = sim.tfidf_cosine()
    print(f"{tfidf_vec_cos=}")
