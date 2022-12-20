"""
Similarity between 2 Corpus.

1. jaccard similarity
2. pca similarity
"""
from scipy.spatial import distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from typing import TYPE_CHECKING, Union

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.freqtable import FreqTable

if TYPE_CHECKING:
    from juxtorpus import Jux


def _cos_sim(v0: Union[np.ndarray, pd.Series], v1: Union[np.ndarray, pd.Series]):
    if isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray):
        assert v0.ndim == 1 and v1.ndim == 1, "Must be 1d array."
        assert v0.shape[0] == v1.shape[0], f"Mismatched shape {v0.shape=} {v1.shape=}"
        if v0.shape[0] == 0: return 0
    elif isinstance(v0, pd.Series) and isinstance(v1, pd.Series):
        assert len(v0) == len(v1), f"Mismatched shape {len(v0)=} {len(v1)=}"
        if len(v0) == 0: return 0
    else:
        raise ValueError(f"They must both be either "
                         f"{np.ndarray.__class__.__name__} or "
                         f"{pd.Series.__class__.__name__}.")
    return 1 - distance.cosine(v0, v1)


class Similarity(object):
    def __init__(self, jux: 'Jux'):
        self._jux = jux

    def jaccard(self, use_lemmas: bool = False):
        """ Return a similarity score between the 2 corpus."""
        if use_lemmas:
            # check if corpus are spacy corpus.
            raise NotImplementedError("To be implemented. Use unique lemmas instead of words.")
        _A_uniqs: set[str] = self._jux.corpus_0.unique_terms
        _B_uniqs: set[str] = self._jux.corpus_1.unique_terms
        return len(_A_uniqs.intersection(_B_uniqs)) / len(_A_uniqs.union(_B_uniqs))

    def lsa_pairwise_cosine(self, n_components: int = 100, verbose=False):
        """ Decompose DTM to SVD and return the pairwise cosine similarity of the right singular matrix.

        Note: this may be different to the typical configuration using a TDM instead of DTM.
        However, sklearn only exposes the right singular matrix.
        tdm.T = (U Sigma V.T).T = V.T.T Sigma.T U.T = V Sigma U.T
        the term-topic matrix of U is now the right singular matrix if we use DTM instead of TDM.
        """
        A, B = self._jux.corpus_0, self._jux.corpus_1
        svd_A = TruncatedSVD(n_components=n_components).fit(A.dtm.tfidf().matrix)
        svd_B = TruncatedSVD(n_components=n_components).fit(B.dtm.tfidf().matrix)
        top_topics = 5
        if verbose:
            top_terms = 5
            for corpus, svd in [(A, svd_A), (B, svd_B)]:
                feature_indices = svd.components_.argsort()[::-1][
                                  :top_topics]  # highest value term in term-topic matrix
                terms = corpus.dtm.term_names[feature_indices]
                for i in range(feature_indices.shape[0]):
                    print(f"Corpus {str(corpus)}: Singular columns [{i}] {terms[i][:top_terms]}")

        # pairwise cosine
        return cosine_similarity(svd_A.components_[:top_topics], svd_B.components_[:top_topics])

    def cosine_similarity(self, metric: str, **kwargs):
        # based on terms
        if metric == 'tf':
            return self._cos_sim_tf(**kwargs)
        elif metric == 'tfidf':
            return self._cos_sim_tfidf(**kwargs)
        elif metric == 'loglikelihood':
            return self._cos_sim_llv(**kwargs)

    def _cos_sim_llv(self, baseline: FreqTable = None):
        if baseline is None:
            baseline = FreqTable.from_freq_tables([self._jux.corpus_0.dtm.freq_table(nonzero=True),
                                                   self._jux.corpus_1.dtm.freq_table(nonzero=True)])

        res = self._jux.stats.log_likelihood_and_effect_size(baseline=baseline).fillna(0)
        return _cos_sim(res['corpus_a_log_likelihood_llv'], res['corpus_b_log_likelihood_llv'])

    def _cos_sim_tf(self, without: list[str] = None) -> float:
        ft_a: FreqTable = self._jux.corpus_0.dtm.freq_table(nonzero=True)
        ft_b: FreqTable = self._jux.corpus_1.dtm.freq_table(nonzero=True)
        if without: ft_a.remove(without), ft_b.remove(without)

        res = pd.concat([ft_a.series, ft_b.series], axis=1).fillna(0)
        return _cos_sim(res[0], res[1])

    def _cos_sim_tfidf(self, **kwargs):
        ft_a: FreqTable = self._jux.corpus_0.dtm.tfidf(**kwargs).freq_table(nonzero=True)
        ft_b: FreqTable = self._jux.corpus_1.dtm.tfidf(**kwargs).freq_table(nonzero=True)
        res = pd.concat([ft_a.series, ft_b.series], axis=1).fillna(0)
        return _cos_sim(res[0], res[1])


if __name__ == '__main__':
    import pandas as pd
    from juxtorpus.corpus import CorpusSlicer
    from juxtorpus.jux import Jux

    corpus = Corpus.from_dataframe(
        pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                    usecols=['processed_text', 'tweet_lga']),
        # pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv',
        #             usecols=['processed_text', 'tweet_lga']),
        col_text='processed_text'
    )

    slicer = CorpusSlicer(corpus)
    brisbane = slicer.filter_by_item('tweet_lga', 'Brisbane (C)')
    fairfield = slicer.filter_by_item('tweet_lga', 'Fairfield (C)')

    sim = Jux(brisbane, fairfield).sim
    pairwise = sim.lsa_pairwise_cosine(n_components=100, verbose=True)
    print(f"SVD pairwise cosine of PCs\n{pairwise}")
    jaccard = sim.jaccard()
    print(f"{jaccard=}")

    sw = nltk.corpus.stopwords.words('english')
    term_vec_cos = sim.cosine_similarity(metric='tf', without=sw + ['climate', 'tweet', 'https'])
    print(f"{term_vec_cos=}")
