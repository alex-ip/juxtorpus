from typing import Union, List, Set, Dict
import pandas as pd
from frozendict import frozendict
from functools import partial

from juxtorpus.meta import Meta, SeriesMeta, DocMeta


class Corpus:
    """ Corpus
    This class wraps around a dataframe of raw str text that represents your corpus.
    It exposes functions that gather statistics on the corpus such as token frequencies and lexical diversity etc.

    summary() provides a quick summary of your corpus.
    """

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_text: str = 'text'):
        meta_df: pd.DataFrame = df.drop(col_text)
        metas: dict[str, SeriesMeta] = dict()
        for col in meta_df.columns:
            # create series meta
            if metas.get(col, None) is None:
                raise KeyError(f"{col} already exists. Please rename the column.")
            metas[col] = SeriesMeta(col, meta_df.loc[:, col])
        return cls(df[col_text], metas)

    COL_TEXT: str = 'text'
    __dtype_text = pd.StringDtype(storage='pyarrow')

    def __init__(self, text: pd.Series, metas: Dict[str, Meta] = None):
        text.name = self.COL_TEXT
        self._df: pd.DataFrame = pd.DataFrame(text, columns=[self.COL_TEXT])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self.COL_TEXT, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_TEXT} column in dataframe."

        # sets the default dtype for texts
        self.__try_text_dtype_conversion(Corpus.__dtype_text)

        # meta data
        self._meta_registry = metas
        if self._meta_registry is None:
            self._meta_registry = dict()

        # processing
        self._processing_history = list()

        # internals
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

    ### Processing ###
    def history(self):
        """ Returns a list of processing history. """
        return self._processing_history.copy()

    def add_process_episode(self, episode):
        self._processing_history.append(episode)

    ### Statistics ###

    @property
    def num_words(self) -> int:
        return self.__num_tokens

    @property
    def num_unique_words(self) -> int:
        return self.__num_uniqs

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def summary(self):
        """ Basic summary statistics of the corpus. """

        #   TODO: compute word statistics W/O any 3rd party dependencies.
        return pd.Series({
            "Number of words": max(self.num_words, 0),
            "Number of unique words": max(self.num_unique_words, 0),
            "Number of documents": len(self)
        }, name='frequency', dtype='uint64')  # supports up to ~4 billion

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

    def cloned(self, mask: 'pd.Series[bool]'):
        """ Returns a clone of itself with the boolean mask applied. """
        text_series = self._df.loc[:, self.COL_TEXT][mask]
        cloned_meta_registry = dict()
        for id_, meta in self._meta_registry.items():
            cloned_meta_registry[id_] = meta.cloned(texts=self._df.loc[:, self.COL_TEXT], mask=mask)
        return Corpus(text_series, cloned_meta_registry)

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __try_text_dtype_conversion(self, dtype):
        try:
            if self._df.loc[:, self.COL_TEXT].dtype != self.__dtype_text:
                self._df = self._df.astype(dtype={self.COL_TEXT: dtype})
        except TypeError:
            print(
                f"[Warn] {self.COL_TEXT} column failed to convert to {dtype} dtype. \
                There will possibly be higher memory consumption however you  may safely ignore this."
            )


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
        super(DummyCorpus, self).__init__(pd.Series(self.dummy_texts), metas=None)


# import aliases
from .builder import CorpusBuilder
from .slicer import CorpusSlicer
