from juxtorpus.corpus import Corpus


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
        print("This extracts the keywords from each corpus, rank them by frequency and display them in a table.")
