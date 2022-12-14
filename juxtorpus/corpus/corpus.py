from typing import Union, Set, Dict, Generator, Optional
import pandas as pd
import spacy.vocab
from spacy.tokens import Doc
from frozendict import frozendict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

from juxtorpus.corpus.meta import MetaRegistry, Meta, SeriesMeta
from juxtorpus.corpus.dtm import DTM
from juxtorpus.matchers import is_word

import logging

logger = logging.getLogger(__name__)


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
            raise ValueError(f"Column {col_text} not found. You must set the col_text argument.\n"
                             f"Available columns: {df.columns}")
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
        self._meta_registry = MetaRegistry(metas)

        # document term matrix - DTM
        self._dtm: Optional[DTM] = DTM()

        # processing
        self._processing_history = list()

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
            root._dtm.build(root.texts())
            # self._dtm.build(root.texts())        # dtm tracks root and builds with root anyway
        return self._dtm

    def find_root(self):
        if self.is_root: return self
        parent = self._parent
        while not parent.is_root:
            parent = parent._parent
        return parent

    ### Meta data ###

    @property
    def meta(self):
        return self._meta_registry.copy()

    def metas(self):
        return self._meta_registry.copy()

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
    def num_terms(self) -> int:
        return self.dtm.total

    @property
    def unique_terms(self) -> set[str]:
        return set(self.dtm.vocab(nonzero=True))

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        docs_info = pd.Series(self.dtm.total_docs_vector).describe()
        # docs_info = docs_info.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        mapper = {row_idx: f"Terms {row_idx}" for row_idx in docs_info.index}
        docs_info.rename(index=mapper, inplace=True)

        other_info = pd.Series({
            "Corpus Type": self.__class__.__name__,
            "Number of documents": len(self),
            "Number of unique terms": len(self.dtm.vocab(nonzero=True)),
        })
        return pd.concat([other_info, docs_info])

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
        return self._dtm.cloned(indices)

    def detached(self):
        """ Detaches from corpus tree and becomes the root.

        DTM will be regenerated when accessed - hence a different vocab.
        """
        self._parent = None
        self._dtm = DTM()
        return self

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc('text')
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]


class SpacyCorpus(Corpus):

    @classmethod
    def from_corpus(cls, corpus: Corpus, docs, vocab):
        return cls(docs, corpus._meta_registry, vocab)

    def __init__(self, docs, metas, nlp: spacy.Language):
        super(SpacyCorpus, self).__init__(docs, metas)
        self._nlp = nlp
        self._is_word_matcher = is_word(self._nlp.vocab)

    @property
    def nlp(self):
        return self._nlp

    @property
    def dtm(self):
        if not self._dtm.is_built:
            root = self.find_root()
            root._dtm.build(root.docs(),
                            vectorizer=CountVectorizer(preprocessor=lambda x: x,
                                                       tokenizer=self._gen_words_from))
        return self._dtm

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT].map(lambda doc: doc.text)

    def docs(self) -> 'pd.Series[Doc]':
        return self._df.loc[:, self.COL_TEXT]

    def _gen_words_from(self, doc):
        return (doc[start: end].text.lower() for _, start, end in self._is_word_matcher(doc))

    def generate_lemmas(self):
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_lemmas_from(texts.iloc[i])

    def _gen_lemmas_from(self, doc):
        return (doc[start: end].lemma_ for _, start, end in self._is_word_matcher(doc))

    def _cloned_texts(self, mask):
        return self.docs().loc[mask]

    def cloned(self, mask: 'pd.Series[bool]'):
        cloned_texts = self._cloned_texts(mask)
        cloned_metas = self._cloned_metas(mask)

        clone = SpacyCorpus(cloned_texts, cloned_metas, self._nlp)
        clone._parent = self

        clone._dtm = self._cloned_dtm(cloned_texts.index)
        clone._processing_history = self._cloned_history()
        return clone
