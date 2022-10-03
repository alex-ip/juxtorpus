"""
Similarity between 2 Corpus.
"""


class Similarity(object):
    def __init__(self, corpus_A, corpus_B):
        self._A = corpus_A
        self._B = corpus_B

    def similarity(self, alg: str):
        """ Return a similarity score between the 2 corpus."""
        if alg.upper() == 'JACCARD':
            _A_uniqs: set[str] = self._A.unique_words()
            _B_uniqs: set[str] = self._B.unique_words()
            return len(_A_uniqs.intersection(_B_uniqs)) / len(_A_uniqs.union(_B_uniqs))
        raise NotImplementedError(f"{alg} is not supported.")
