""" Corpus
The data model for a corpus. Contains basic summary statistics.

You may ingest and extract data from and to its persisted form. (e.g. csv)
"""
from datetime import datetime
from typing import Union, List, Dict, Set
import pandas as pd
import string, re
import time
import spacy
import weakref

from juxtorpus import nlp
from juxtorpus.matchers import no_puncs


class CorpusMeta:
    """ CorpusMeta
    stores the metadata of the corpus and loads them lazily when required.
    But this only works with disk? -- too many responsibilities. Scrap lazy loading?
    """

    def __init__(self, df_meta: pd.DataFrame):
        self._df_meta = df_meta


class Corpus:
    """ Corpus
    This class wraps around a dataframe of raw str text that represents your corpus.
    It exposes functions that gather statistics on the corpus such as token frequencies and lexical diversity etc.

    summary() provides a quick summary of your corpus.

    Some caveats (mostly involving implicit internal states)
    + __doc__ column is maintained by this object. Do not try to change data in this column.
    """

    @staticmethod
    def from_disk(path: str, sep=',') -> 'Corpus':
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
            return Corpus(df=pd.read_csv(path, sep=sep))
        raise Exception("Corpus currently only supports .csv formats.")

    @staticmethod
    def from_(texts: Union[List[str], Set[str]]) -> 'Corpus':
        if isinstance(texts, list) or isinstance(texts, set):
            return Corpus(df=pd.DataFrame(texts, columns=[Corpus.COL_TEXT]))
        raise Exception("Corpus currently only supports lists and sets.")

    def to(self, type_: str):
        if type_ == 'csv':
            raise NotImplemented("Exports to csv.")
        raise NotImplemented()

    COL_TEXT: str = 'text'
    COL_DOC: str = '__doc__'  # spacy Document
    dtype_text = pd.StringDtype(storage='pyarrow')

    def __init__(self, df: pd.DataFrame):

        self._df: pd.DataFrame = df
        if self.COL_TEXT not in self._df.columns:
            raise ValueError(f"Missing {self.COL_TEXT} column in dataframe.")
        assert len(list(filter(lambda x: x == self.COL_TEXT, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_TEXT} column in dataframe."

        # 1. sets the default dtype for texts
        try:
            self._df[self.COL_TEXT] = self._df[self.COL_TEXT].astype(dtype=self.dtype_text)
        except Exception:
            raise TypeError(f"{self.COL_TEXT} failed to convert to string dtype.")

        # todo: 2. build the meta corpus if exists.

        self._num_tokens: int = -1
        self._num_uniqs: int = -1

    def preprocess(self, verbose: bool = False):
        start = time.time()
        if verbose: print(f"++ Preprocessing {len(self._df)} documents...")

        if self.COL_DOC in self._df.columns:
            return self

        if len(self._df) < 100:
            self._df[self.COL_DOC] = self._df[self.COL_TEXT].apply(lambda x: nlp(x))
        else:
            self._df[self.COL_DOC] = list(nlp.pipe(self._df[self.COL_TEXT]))
        if verbose: print(f"++ Done. Elapsed: {time.time() - start}")
        return self

    @property
    def num_tokens(self) -> int:
        self._compute_word_statistics()
        return self._num_tokens

    @property
    def num_uniq_tokens(self) -> int:
        self._compute_word_statistics()
        return self._num_uniqs

    @property
    def df(self):
        # TODO: join the dataframe with the metadata and return a COPY of the dataframe. Perhaps drop doc?
        return None

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def docs(self) -> 'pd.Series[spacy.tokens.doc.Doc]':
        return self._df.loc[:, self.COL_DOC]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        self._compute_word_statistics()
        return pd.Series({
            "Number of words": self.num_tokens,
            "Number of unique words": self.num_uniq_tokens,
            "Number of documents": len(self)
        })

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

    def groupby(self, columns: Union[str, List[str]]):
        """ Returns a dictionary of corpus grouped by the column."""
        # dev notes: it doesn't support the split, apply and join workflow like in pandas but since we only
        # expect text data, this workflow is unlikely? to be used.

        # 1. load the appropriate metadata column from the data.
        groups = self._df.groupby(columns)

        corpus_groups = dict()
        for cat_name, group_df in groups:
            corpus_groups[cat_name] = Corpus(df=group_df)
        return FrozenCorpusGroups(weakref.ref(self), corpus_groups)

    def _compute_word_statistics(self):
        if Corpus.COL_DOC not in self._df.columns:
            raise RuntimeError("You need to call preprocess() on your corpus object first.")

        if self._num_tokens > -1 or self._num_uniqs > -1:
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

            self._num_tokens = _num_tokens
            self._num_uniqs = len(_uniqs)

    def __len__(self):
        return len(self._df) if self._df is not None else 0


class CorpusGroups(dict):
    def join(self):
        pass  # do alignment of dict if no original corpus?
        # todo: join corpus - perhaps expose a .join(corpus) method in the Corpus class.


class FrozenCorpusGroups(CorpusGroups):
    """ Immutable corpus groups
    This class is used to return the result of a groupby call from a corpus.
    """

    def __init__(self, orig_corpus: weakref.ReferenceType[Corpus], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_corpus = orig_corpus

    def join(self) -> Union[Corpus, None]:
        """ This returns the original corpus where they were grouped from.
        Caveat: if no hard references to the original corpus is kept, this returns None.

        This design was chosen as we expect the user to reference the original corpus themselves
        instead of calling join().
        """
        return self._original_corpus()  # returns the hard reference of the weakref.

    def __setitem__(self, key, value):
        raise RuntimeError("You may not write to FrozenCorpusGroups.")


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
        super(DummyCorpus, self).__init__(df=pd.DataFrame(self.dummy_texts, columns=[Corpus.COL_TEXT]))


if __name__ == '__main__':
    # trump = Corpus.from_disk("~/Downloads/2017_01_18_trumptweets.csv")
    trump = Corpus.from_disk("../assets/samples/tweetsA.csv")
    print(trump.texts()[0])
    trump.preprocess()
    print(trump.summary())
