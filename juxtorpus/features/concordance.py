import weakref
from abc import ABCMeta, abstractmethod
from atap_widgets.concordance import ConcordanceTable, ConcordanceWidget
from juxtorpus.corpus import Corpus
import pandas as pd
from typing import Union


class Concordance(metaclass=ABCMeta):
    """ Concordance
    This is the base concordance class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def set_keyword(self, keyword: str):
        return self

    @abstractmethod
    def find(self):
        raise NotImplementedError()


class ATAPConcordance(ConcordanceWidget, Concordance):
    """ ATAPConcordance

    This implementation integrates with the Concordance tool from atap_widgets package.


    Usage:
    ```
    from juxtorpus.corpus import DummyCorpus
    c = ATAPConcordance(corpus=DummyCorpus().preprocess())
    print(c.set_keyword('Australia').find())
    ```
    """

    def __init__(self, corpus: Corpus):
        corpus_df = corpus._df.rename(columns={Corpus.COL_DOC: 'spacy_doc'})
        if Corpus.COL_DOC not in corpus._df.columns:
            raise RuntimeError("Corpus is not preprocessed.")
        super(ATAPConcordance, self).__init__(df=corpus_df)  # builds ConcordanceTable internally.

        # perf:
        self._keyword_prev: str = ''
        self._results_cache: Union[pd.DataFrame, None] = None

    def set_keyword(self, keyword: str):
        self._keyword_prev = self.search_table.keyword
        if keyword == self.search_table.keyword:
            return self
        self.search_table.keyword = keyword
        return self

    def find(self) -> pd.DataFrame:
        if len(self.search_table.keyword) < 1:
            raise ValueError("Did you set the keyword? Call set_keyword()")

        if self._keyword_updated():
            self._results_cache = self.search_table._get_results()
        return self._results_cache

    def show_widget(self):
        return self.show()

    def _keyword_updated(self):
        return self._keyword_prev != self.search_table.keyword


if __name__ == '__main__':
    from juxtorpus.corpus import DummyCorpus

    atap_concordance = ATAPConcordance(corpus=DummyCorpus().preprocess())
    atap_concordance.set_keyword('australia').find()
    atap_concordance.set_keyword('reading').find()
