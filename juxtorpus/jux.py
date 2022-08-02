from juxtorpus.corpus import Corpus
from juxtorpus.features.keywords import Keywords, RakeKeywords, TFKeywords, TFIDFKeywords

from typing import Tuple, List
import pandas as pd


class Jux:
    """ Jux
    This is the main class for Juxtorpus. It takes in 2 corpus and exposes numerous functions
    to help contrast the two corpus.

    It is expected that the exposed functions are used as tools that serve as building blocks
    for your own further analysis.
    """

    def __init__(self, corpusA: Corpus, corpusB: Corpus):
        self._A = corpusA
        self._B = corpusB

    @property
    def corpusA(self):
        return self._A

    @property
    def corpusB(self):
        return self._B

    def keywords(self, method: str):
        """ Extract and return the keywords of the two corpus ranked by frequency. """
        _extractor: Keywords
        if method == 'rake':
            _extractor = RakeKeywords(corpusA=self._A, corpusB=self._B)
            kw = _extractor.extracted()

            a = pd.DataFrame(kw[0], columns=['A_keyphrase', 'A_freq'])
            b = pd.DataFrame(kw[1], columns=['B_keyphrase', 'B_freq'])
        elif method == 'freq':
            _extractor = TFKeywords(corpusA=self._A, corpusB=self._B)
            kw = _extractor.extracted()
            a = pd.DataFrame(kw[0], columns=['A_keyphrase', 'A_freq_normalised'])
            b = pd.DataFrame(kw[1], columns=['B_keyphrase', 'B_freq_normalised'])

        elif method == 'tfidf':
            _extractor = TFIDFKeywords(corpusA=self._A, corpusB=self._B)
            kw = _extractor.extracted()
            a = pd.DataFrame(kw[0], columns=['A_keyphrase', 'A_tfidf'])
            b = pd.DataFrame(kw[1], columns=['B_keyphrase', 'B_tfidf'])
        else:
            raise ValueError("Unsupported keyword extraction method.")
        # use sets to compare
        return pd.concat([a, b], axis=1)

    def lexical_diversity(self):
        # number of uniq / number of words      of course this will be a problem if the 2 corpus have different sizes. So maybe this is a log relationship.
        pass

    def distance(self):
        print("This calculates the distance between the corpora.")


class JuxAssistant(Jux):
    """
    JuxAssistant functions as a wrapper class on Jux but is a child of Jux for better flexibility to the user.

    This class expose numerous function representative of different use cases using Jux functions as building blocks.
    It'll try to impose best practices in forms of default values and printed warnings. List as follows:
    1. using normalised frequencies instead of raw
    2. check size of corpus to justify a frequency analysis (~ > 1 million words)
    3.
    """

    def use_case_A(self):
        pass
