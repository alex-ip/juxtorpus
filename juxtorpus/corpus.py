""" Corpus
The data model for a corpus. Contains basic summary statistics.

You may ingest and extract data from and to its persisted form. (e.g. csv)
"""
from datetime import datetime
from typing import Union, List
import pandas as pd


class Corpus:

    @staticmethod
    def from_(path: str, sep=',') -> 'Corpus':
        """
        Ingest data and return the Corpus data model.
        :param path: Path to csv
        :param sep: csv separator
        :return:
        """
        if path.endswith('csv'):
            return Corpus(docs=pd.read_csv(path, sep=sep))
        raise Exception("Currently only supports .csv formats.")

    @staticmethod
    def to(type_: str) -> 'Corpus':
        pass

    def __init__(self, docs: Union[List[str], pd.DataFrame], datetime_: datetime = None):
        self._datetime = datetime_  # metadata - date and time  TODO: need to rethink how to add these meta data
        self._para_split = '\n'
        self._doc_col_name = 'doc'

        self.docs: pd.DataFrame
        if isinstance(docs, List):
            self.docs = pd.DataFrame(docs, columns=[self._doc_col_name])
        elif isinstance(docs, pd.DataFrame):
            self.docs = docs
            if self._doc_col_name not in self.docs.columns:
                raise ValueError(f"Missing {self._doc_col_name} column in dataframe.")
        else:
            raise ValueError("Docs must either be a list of string or a pandas dataframe.")

    # TODO: write the paragraph function
    def paragraphs(self, split=None):
        """ Split the corpus into paragraphs from all documents. """
        _split = self._para_split
        if split is not None:
            _split = split
            # TODO: regenerate cache if needed.

    def unique_tokens(self):
        # TODO: cache a set of the unique tokens
        pass

    def summary(self):
        """ Basic summary statistics of the corpus. """
        # TODO: number of tokens
        # TODO: number of unique tokens
        # TODO: number of tokens (after stemming)
        # TODO: number of unique tokens (after stemming)
        # TODO: number of tokens (after lemmatisation)
        # TODO: number of unique tokens (after lemmatisation) - we will need our own lemma dict
        # TODO: word frequency distribution of top 10 words
        # TODO: word frequency distribution of top 10 stemmed words
        # TODO: word frequency distribution of top 10 lemmatised words

    def freq_of(self, word: str, normalised: bool = False):
        """ Returns the frequency of the word. """
        # TODO: frequency of word in corpus
        # TODO: frequency of stemmed word in stemmed corpus?
        # TODO: normalised frequency of word
        # TODO: normalised frequency of stemmed word in stemmed corpus.
        pass

    def _num_tokens(self) -> int:
        # TODO: return the number of tokens in the corpus
        pass

    def _num_unique_tokens(self):
        pass

    def __len__(self):
        return len(self.docs) if self.docs is not None else 0
