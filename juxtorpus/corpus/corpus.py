from typing import Union, Set, Dict, Generator, Optional
import pandas as pd
import spacy.vocab
from frozendict import frozendict
from collections import Counter
import re

from juxtorpus.corpus.meta import Meta, SeriesMeta
from juxtorpus.corpus.dtm import DTM


class Corpus:
    """ Corpus
    This class wraps around a dataframe of raw str text that represents your corpus.
    It exposes functions that gather statistics on the corpus such as token frequencies and lexical diversity etc.

    summary() provides a quick summary of your corpus.
    """

    COL_TEXT: str = 'text'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_text: str = COL_TEXT):
        if col_text not in df.columns:
            raise ValueError(f"Column {col_text} not found. You may set the col_text argument.")
        meta_df: pd.DataFrame = df.drop(col_text, axis=1)
        metas: dict[str, SeriesMeta] = dict()
        for col in meta_df.columns:
            # create series meta
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please rename the column.")
            metas[col] = SeriesMeta(col, meta_df.loc[:, col])
        return Corpus(df[col_text], metas)

    def __init__(self, text: pd.Series,
                 metas: Dict[str, Meta] = None):
        text.name = self.COL_TEXT
        self._df: pd.DataFrame = pd.DataFrame(text, columns=[self.COL_TEXT])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self.COL_TEXT, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_TEXT} column in dataframe."

        self._parent: Optional[Corpus] = None

        # meta data
        self._meta_registry = metas if metas is not None else dict()

        # document term matrix - DTM
        self._dtm: Optional[DTM] = DTM()

        # processing
        self._processing_history = list()

        # internals - word statistics
        self._counter: Union[Counter[str, int], None] = None
        self._num_tokens: int = -1
        self._num_words: int = -1
        self._num_uniqs: int = -1

    @property
    def parent(self):
        return self._parent

    @property
    def is_root(self):
        return self._parent is None

    ### Document Term Matrix ###
    @property
    def dtm(self):
        if not self._dtm.is_built:
            root = self.find_root()
            root._dtm.build(root.generate_words())
        return self._dtm

    def find_root(self):
        if self.is_root: return self
        parent = self._parent
        while not parent.is_root:
            parent = parent._parent
        return parent

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
    def num_tokens(self) -> int:
        if self._num_tokens < 0:
            self._num_tokens = sum((len(list(token_list)) for token_list in self.generate_tokens()))
        return self._num_tokens

    @property
    def num_words(self) -> int:
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return self._num_words

    @property
    def num_unique_words(self) -> int:
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return self._num_uniqs

    @property
    def unique_words(self) -> set[str]:
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return set(self._counter.keys())

    def word_counter(self) -> Counter:
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return self._counter.copy()

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return pd.Series({
            "Number of words": max(self.num_words, 0),
            "Number of unique words": max(self.num_unique_words, 0),
            "Number of documents": len(self)
        }, name='frequency', dtype='uint64')  # supports up to ~4 billion

    def freq_of(self, words: Set[str]):
        """ Returns the frequency of a list of words. """
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        freqs = dict()
        if isinstance(words, str):
            freqs[words] = self._counter.get(words, 0)
            return freqs
        else:
            for word in words:
                freqs[word] = self._counter.get(word, 0)

    def most_common_words(self, n: int):
        if not self._computed_word_statistics():
            self._compute_word_statistics()
        return self._counter.most_common(n)

    def _computed_word_statistics(self):
        return self._counter is not None

    def _compute_word_statistics(self):
        self._counter = Counter()
        self.texts().apply(lambda text: self._counter.update(self._gen_words_from(text)))
        self._num_words = sum(self._counter.values())  # total() may be used for python >3.10
        self._num_uniqs = len(self._counter.keys())

    def generate_words(self):
        """ Generate list of words for each document in the corpus. """
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_words_from(texts.iloc[i])

    def _gen_words_from(self, text) -> Generator[str, None, None]:
        return (token.lower() for token in re.findall('[A-Za-z]+', text))

    def generate_tokens(self):
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_tokens_from(texts.iloc[i])

    def _gen_tokens_from(self, text) -> Generator[str, None, None]:
        return (token.lower() for token in text.split(" "))

    def cloned(self, mask: 'pd.Series[bool]'):
        """ Returns a (usually smaller) clone of itself with the boolean mask applied. """
        cloned_texts = self._cloned_texts(mask)
        cloned_metas = self._cloned_metas(mask)

        clone = Corpus(cloned_texts, cloned_metas)
        clone._parent = self

        clone._dtm = self._cloned_dtm(cloned_texts.index)
        clone._processing_history = self._cloned_history()
        return clone

    def _cloned_texts(self, mask):
        return self.texts()[mask]

    def _cloned_metas(self, mask):
        cloned_meta_registry = dict()
        for id_, meta in self._meta_registry.items():
            cloned_meta_registry[id_] = meta.cloned(texts=self._df.loc[:, self.COL_TEXT], mask=mask)
        return cloned_meta_registry

    def _cloned_history(self):
        return [h for h in self.history()]

    def _cloned_dtm(self, indices):
        return self.dtm.cloned(self.dtm, indices)

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc('text')
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]


from juxtorpus.matchers import is_word


class SpacyCorpus(Corpus):

    @classmethod
    def from_corpus(cls, corpus: Corpus, docs, vocab):
        return cls(docs, corpus._meta_registry, vocab)

    def __init__(self, docs, metas, vocab: spacy.vocab.Vocab):
        super(SpacyCorpus, self).__init__(docs, metas)
        self._vocab = vocab
        self._is_word_matcher = is_word(vocab)

    @property
    def vocab(self):
        return self._vocab

    def _gen_words_from(self, text):
        return (text[start: end].text.lower() for _, start, end in self._is_word_matcher(text))

    def cloned(self, mask: 'pd.Series[bool]'):
        cloned_texts = self._cloned_texts(mask)
        cloned_metas = self._cloned_metas(mask)

        clone = SpacyCorpus(cloned_texts, cloned_metas, self._vocab)
        clone._parent = self

        clone._dtm = self._cloned_dtm(cloned_texts.index)
        clone._processing_history = self._cloned_history()
        return clone
