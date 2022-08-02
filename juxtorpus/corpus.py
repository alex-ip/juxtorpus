""" Corpus
The data model for a corpus. Contains basic summary statistics.

You may ingest and extract data from and to its persisted form. (e.g. csv)
"""
from datetime import datetime
from typing import Union, List, Dict
import pandas as pd
import string, re


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

    @staticmethod
    def to(type_: str) -> 'Corpus':
        pass

    def __init__(self, docs: Union[List[str], pd.DataFrame], datetime_: datetime = None):
        self._datetime = datetime_  # metadata - date and time  TODO: need to rethink how to add these meta data
        self._para_split = '\n'
        self._doc_col_name = 'doc'

        self._docs_df: pd.DataFrame
        if isinstance(docs, List):
            self._docs_df = pd.DataFrame(docs, columns=[self._doc_col_name])
        elif isinstance(docs, pd.DataFrame):
            self._docs_df = docs
            if self._doc_col_name not in self._docs_df.columns:
                raise ValueError(f"Missing {self._doc_col_name} column in dataframe.")
        else:
            raise ValueError("Docs must either be a list of string or a pandas dataframe.")

        try:
            self._docs_df[self._doc_col_name] = self._docs_df[self._doc_col_name].astype(dtype=str)
        except Exception:
            raise TypeError("doc column must be string.")

        self._token_stats_cache: Dict[str, int] = None

    @property
    def docs(self) -> List[str]:
        return self._docs_df[self._doc_col_name].tolist()

    @property
    def num_tokens(self) -> int:
        return self._tokens_statistics().get("num_tokens")

    @property
    def num_uniq_tokens(self) -> int:
        return self._tokens_statistics().get("num_uniques")

    # TODO: write the paragraph function
    def paragraphs(self, split=None):
        """ Split the corpus into paragraphs from all documents. """
        _split = self._para_split
        if split is not None:
            _split = split
            # TODO: regenerate cache if needed.

    def summary(self):
        """ Basic summary statistics of the corpus. """
        token_stats: Dict[str, int] = self._tokens_statistics()
        return pd.Series({
            "Number of words": token_stats.get("num_tokens"),
            "Number of unique words": token_stats.get("num_uniques")
        })

    def freq_of(self, words: List[str], normalised: bool = False):
        """ Returns the frequency of the word. """
        word_dict = dict()
        for w in words:
            word_dict[w] = 0
        for i in range(len(self._docs_df)):
            _doc = self._docs_df[self._doc_col_name].iloc[i]
            _tokens = Corpus._preprocess(_doc).split()
            for t in _tokens:
                if word_dict.get(t, None) is not None:
                    word_dict[t] += 1

    def _tokens_statistics(self) -> Dict[str, int]:

        if self._token_stats_cache is not None:
            return self._token_stats_cache
        else:
            _num_tokens: int = 0
            _uniqs = set()
            for i in range(len(self._docs_df)):
                _doc = self._docs_df[self._doc_col_name].iloc[i]
                _tokens = Corpus._preprocess(_doc).split()
                _num_tokens += len(_tokens)
                for t in _tokens:
                    _uniqs.add(t)
            return {
                "num_tokens": _num_tokens,
                "num_uniques": len(_uniqs)
            }




    def __len__(self):
        return len(self._docs_df) if self._docs_df is not None else 0

    @staticmethod
    def _preprocess(text: str):
        text = text.lower()
        text = Corpus._remove_punctuations(text)
        return text

    @staticmethod
    def _remove_punctuations(s: str):
        to_remove = string.punctuation
        return re.sub(f"[{to_remove}]", '', s, count=0)


class DummyCorpus(Corpus):
    dummy_texts = [
        "Hello, this is a dummy text 1.",
        "Hello, this is another dummy text.",
        "Hello, this is yet another dummy text."
    ]

    def __init__(self):
        super(DummyCorpus, self).__init__(docs=DummyCorpus.dummy_texts)


class DummyCorpusB(DummyCorpus):
    dummy_texts = [
        "This is a dummy corpus to be compared to.",
        "This is another dummy corpus to be compared to."
    ]

    def __init__(self):
        super(DummyCorpus, self).__init__(docs=DummyCorpusB.dummy_texts)


if __name__ == '__main__':
    trump = Corpus.from_("~/Downloads/2017_01_18_trumptweets.csv")
    print(trump.docs[0])
    print(trump.summary())
