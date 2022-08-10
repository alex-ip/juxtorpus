""" Corpus
The data model for a corpus. Contains basic summary statistics.

You may ingest and extract data from and to its persisted form. (e.g. csv)
"""
from datetime import datetime
from typing import Union, List, Dict
import pandas as pd
import string, re
import time
import spacy

from juxtorpus import nlp
from juxtorpus.matchers import no_puncs


class CorpusMeta:
    def __init__(self):
        self.preprocessed_elapsed_time = None


class Corpus:

    @staticmethod
    def from_(path: str, sep=',') -> 'Corpus':
        """
        Ingest data and return the Corpus data model.
        :param path: Path to csv
        :param sep: csv separator
        :return:
        """
        # TODO: accept .txt file (likely most typical form of corpus storage)
        if path.endswith('.txt'):
            raise NotImplemented(".txt file not implemented yet.")

        if path.endswith('.csv'):
            return Corpus(docs=pd.read_csv(path, sep=sep))
        raise Exception("Corpus currently only supports .csv formats.")

    def to(self, type_: str):
        if type_ == 'csv':
            raise NotImplemented("Exports to csv.")
        pass

    def __init__(self, docs: Union[List[str], pd.DataFrame]):
        self._col_text = 'text'
        self._col_doc = 'doc'  # spacy Document
        self._meta = CorpusMeta()  # corpus meta data

        self._df: pd.DataFrame
        if isinstance(docs, list):
            self._df = pd.DataFrame(docs, columns=[self._col_text])
        elif isinstance(docs, pd.DataFrame):
            self._df = docs
            if self._col_text not in self._df.columns:
                raise ValueError(f"Missing {self._col_text} column in dataframe.")
        else:
            raise ValueError("Docs must either be a list of string or a pandas dataframe.")

        try:
            self._df[self._col_text] = self._df[self._col_text].astype(dtype=str)
        except Exception:
            raise TypeError(f"{self._col_text} column must be string.")

        self._word_stats_cache: Dict[str, int] = None

    def preprocess(self, verbose: bool = False):
        start = time.time()
        if verbose: print(f"++ Preprocessing {len(self._df)} documents...")

        if len(self._df) < 100:
            self._df[self._col_doc] = self._df[self._col_text].apply(lambda x: nlp(x))
        else:
            self._df[self._col_doc] = list(nlp.pipe(self._df[self._col_text]))
        if verbose: print(f"++ Done. Elapsed: {time.time() - start}")
        return self

    @property
    def texts(self) -> List[str]:
        return self._df[self._col_text].tolist()

    @property
    def docs(self) -> List[str]:
        return self._df[self._col_doc].tolist()

    @property
    def num_tokens(self) -> int:
        return self._word_statistics().get("num_tokens")

    @property
    def num_uniq_tokens(self) -> int:
        return self._word_statistics().get("num_uniques")

    def summary(self):
        """ Basic summary statistics of the corpus. """
        token_stats: Dict[str, int] = self._word_statistics()
        return pd.Series({
            "Number of words": token_stats.get("num_tokens"),
            "Number of unique words": token_stats.get("num_uniques")
        })

    def freq_of(self, words: List[str], normalised: bool = False):
        """ Returns the frequency of a list of words. """
        word_dict = dict()
        for w in words:
            word_dict[w] = 0
        for i in range(len(self._df)):
            _doc = self._df[self._col_text].iloc[i]
            for t in _doc:
                if word_dict.get(t, None) is not None:
                    word_dict[t] += 1

    def _word_statistics(self) -> Dict[str, int]:

        if self._word_stats_cache is not None:
            return self._word_stats_cache
        else:
            _num_tokens: int = 0
            _uniqs = set()
            for i in range(len(self._df)):
                _doc = self._df[self._col_doc].iloc[i]
                _no_puncs_doc = no_puncs(nlp.vocab)(_doc)
                _num_tokens += len(_no_puncs_doc)
                for t in _no_puncs_doc:
                    _uniqs.add(t)
            return {
                "num_tokens": _num_tokens,
                "num_uniques": len(_uniqs)
            }

    def __len__(self):
        return len(self._df) if self._df is not None else 0


class DummyCorpus(Corpus):
    dummy_texts = [
        "The cafe is empty aside from an old man reading a book about Aristotle."
        "In Australia, Burger King is called Hungry Jacks.",
        "She is poor but quite respectable.",
        "She was very tired and frustrated.",
        "What's your address?",
        "Man it is hot in Australia!"
    ]

    def __init__(self):
        super(DummyCorpus, self).__init__(docs=DummyCorpus.dummy_texts)


if __name__ == '__main__':
    # trump = Corpus.from_("~/Downloads/2017_01_18_trumptweets.csv")
    trump = Corpus.from_("assets/samples/tweetsA.csv")
    print(trump.docs[0])
    print(trump.preprocess())
    print(trump.summary())
