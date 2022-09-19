""" Corpus
The data model for a corpus. Contains basic summary statistics.

You may ingest and extract data from and to its persisted form. (e.g. csv)
"""
from typing import Union, List, Set, Dict
import pandas as pd
import time
from spacy.tokens import Doc
from frozendict import frozendict

from juxtorpus import nlp
from juxtorpus.matchers import no_puncs
from juxtorpus.meta import Meta


class Corpus:
    """ Corpus
    This class wraps around a dataframe of raw str text that represents your corpus.
    It exposes functions that gather statistics on the corpus such as token frequencies and lexical diversity etc.

    summary() provides a quick summary of your corpus.

    Some caveats (mostly involving implicit internal states)
    + __doc__ column is maintained by this object. Do not try to change data in this column.
    """

    COL_TEXT: str = 'text'
    COL_DOC: str = '__doc__'  # spacy Document
    __dtype_text = pd.StringDtype(storage='pyarrow')

    def __init__(self, text: pd.Series, metas: Dict[str, Meta] = None):

        self._df: pd.DataFrame = pd.DataFrame(text)
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self.COL_TEXT, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_TEXT} column in dataframe."

        # sets the default dtype for texts
        try:
            if self._df.loc[:, self.COL_TEXT].dtype != self.__dtype_text:
                self._df = self._df.astype(dtype={self.COL_TEXT: self.__dtype_text})
        except Exception:
            print(f"[Warn] {self.COL_TEXT} column failed to convert to pyarrow dtype. You may ignore this.")
            pass

        # meta data
        self._meta_registry = metas
        if self._meta_registry is None:
            self._meta_registry = dict()

        # internals properties
        self.__num_tokens: int = -1
        self.__num_uniqs: int = -1

    ### Meta data ###

    def metas(self):
        return frozendict(self._meta_registry)

    def add_meta(self, meta: Meta):
        if self._meta_registry.get(meta.id, None) is not None:
            raise ValueError("Meta id must be unique. Try calling .metas() method to view existing metas.")
        self._meta_registry[meta.id] = meta

    def remove_meta(self, id_: str):
        del self._meta_registry[id_]

    def get_meta(self, id_: str):
        return self._meta_registry.get(id_, None)

    ### Preprocessing ###

    def preprocess(self, verbose: bool = False):
        start = time.time()
        if verbose: print(f"++ Preprocessing {len(self._df)} documents...")

        if self.is_processed:
            return self

        if len(self._df) < 100:
            self._df[self.COL_DOC] = self._df[self.COL_TEXT].apply(lambda x: nlp(x))
        else:
            self._df[self.COL_DOC] = list(nlp.pipe(self._df[self.COL_TEXT]))
        if verbose: print(f"++ Done. Elapsed: {time.time() - start}")
        return self

    @property
    def is_processed(self):
        return self.COL_DOC in self._df.columns

    @property
    def num_words(self) -> int:
        self._compute_word_statistics()
        return self.__num_tokens

    @property
    def num_unique_words(self) -> int:
        self._compute_word_statistics()
        return self.__num_uniqs

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def docs(self) -> 'pd.Series[Doc]':
        return self._df.loc[:, self.COL_DOC]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        self._compute_word_statistics()
        return pd.Series({
            "Number of words": self.num_words,
            "Number of unique words": self.num_unique_words
        }, name='frequency')

    def freq_of(self, words: Set[str]):
        """ Returns the frequency of a list of words. """
        word_dict = dict()
        for w in words:
            word_dict[w] = 0
        for i in range(len(self._df)):
            _doc = self._df[self.COL_TEXT].iloc[i]
            for t in _doc:
                if word_dict.get(t, None) is not None:
                    word_dict[t] += 1
        return word_dict

    def _compute_word_statistics(self):
        if Corpus.COL_DOC not in self._df.columns:
            raise RuntimeError("You need to call preprocess() on your corpus object first.")

        if self.__num_tokens > -1 or self.__num_uniqs > -1:
            pass
        else:
            _num_tokens: int = 0
            _uniqs = set()
            _no_puncs = no_puncs(nlp.vocab)
            for i in range(len(self._df)):
                _doc = self._df[self.COL_DOC].iloc[i]
                _no_puncs_doc = _no_puncs(_doc)
                _num_tokens += len(_no_puncs_doc)
                for _, start, end in _no_puncs_doc:
                    _uniqs.add(_doc[start:end].text.lower())

            self.__num_tokens = _num_tokens
            self.__num_uniqs = len(_uniqs)

    def __len__(self):
        return len(self._df) if self._df is not None else 0


class TweetCorpus(Corpus):
    # the features/attributes is a superset of corpus.
    pass


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
        super(DummyCorpus, self).__init__(df=pd.DataFrame(self.dummy_texts, columns=[Corpus.COL_TEXT]), metas=None)


if __name__ == '__main__':
    pass
