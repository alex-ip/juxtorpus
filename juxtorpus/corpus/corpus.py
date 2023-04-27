from typing import Dict, Generator, Optional, Callable, Union
import pandas as pd
import spacy.vocab
from spacy.tokens import Doc
from spacy import Language
from sklearn.feature_extraction.text import CountVectorizer
import re
import coolname

from juxtorpus.interfaces.clonable import Clonable
from juxtorpus.corpus.slicer import CorpusSlicer, SpacyCorpusSlicer
from juxtorpus.corpus.meta import Meta, SeriesMeta
from juxtorpus.corpus.dtm import DTM
from juxtorpus.corpus.viz import CorpusViz
from juxtorpus.matchers import is_word, is_word_tweets, is_hashtag, is_mention

import logging

logger = logging.getLogger(__name__)

TDoc = Union[str, Doc]

_ALL_CORPUS_NAMES = set()
_CORPUS_NAME_SEED = 42


def generate_name(corpus: 'Corpus') -> str:
    # todo: should generate a random name based on corpus words
    # tmp solution - generate a random name.
    while name := coolname.generate_slug(2):
        if name in _ALL_CORPUS_NAMES:
            continue
        else:
            return name


class Corpus(Clonable):
    """ Corpus
    This class abstractly represents a corpus which is a collection of documents.
    Each document is also described by their metadata and is used for functions such as slicing.

    An important component of the Corpus is that it also holds the document-term matrix which you can access through
    the accessor `.dtm`. See class DTM. The dtm is lazily loaded and is always computed for the root corpus.
    (read further for a more detailed explanation.)

    A main design feature of the corpus is to allow for easy slicing and dicing based on the associated metadata,
    text in document. See class CorpusSlicer. After each slicing operation, new but sliced Corpus object is
    returned exposing the same descriptive functions (e.g. summary()) you may wish to call again.

    To build a corpus, use the CorpusBuilder. This class handles the complexity

    ```
    builder = CorpusBuilder(pathlib.Path('./data.csv'))
    builder.add_metas('some_meta', 'datetime')
    builder.set_text_column('text')
    corpus = builder.build()
    ```

    Internally, documents are stored as rows of string in a dataframe. Metadata are stored in the meta registry.
    Slicing is equivalent to creating a `cloned()` corpus and is really passing a boolean mask to the dataframe and
    the associated metadata series. When sliced, corpus objects are created with a reference to its parent corpus.
    This is mainly for performance reasons, so that the expensive DTM computed may be reused and a shared vocabulary
    is kept for easier analysis of different sliced sub-corpus. You may choose the corpus to be `detached()` from this
    behaviour, and the corpus will act as the root, forget its lineage and a new dtm will need to be rebuilt.
    """

    class DTMRegistry(dict):
        def __init__(self, *args, **kwargs):
            super(Corpus.DTMRegistry, self).__init__(*args, **kwargs)
            self.set_tokens_dtm(DTM())

        def __setitem__(self, key, value):
            if not isinstance(value, DTM):
                raise ValueError(f"{self.__class__.__name__} only holds {DTM.__name__} objects.")
            super(Corpus.DTMRegistry, self).__setitem__(key, value)

        def set_tokens_dtm(self, dtm: DTM):
            self['tokens'] = dtm

        def get_tokens_dtm(self) -> DTM:
            return self.get('tokens')

        def set_custom_dtm(self, dtm):
            self['custom'] = dtm

        def get_custom_dtm(self) -> DTM:
            return self.get('custom', None)

    class MetaRegistry(dict):
        def __init__(self, *args, **kwargs):
            super(Corpus.MetaRegistry, self).__init__(*args, **kwargs)

        def __setitem__(self, key, value):
            if not isinstance(value, Meta): raise ValueError(f"MetaRegistry only holds {Meta.__name__} objects.")
            super(Corpus.MetaRegistry, self).__setitem__(key, value)

        def summary(self):
            """ Returns a summary of the metadata information. """
            infos = (meta.summary() for meta in self.values())
            df = pd.concat(infos, axis=0).fillna('')

            return df.T

        def get_or_raise_err(self, id_: str):
            meta = self.get(id_, None)
            if meta is None: raise KeyError(f"{id_} does not exist.")
            return meta

    COL_DOC: str = 'document'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_doc: str = COL_DOC, name: str = None) -> 'Corpus':
        if col_doc not in df.columns:
            raise ValueError(f"Column {col_doc} not found. You must set the col_doc argument.\n"
                             f"Available columns: {df.columns}")
        meta_df: pd.DataFrame = df.drop(col_doc, axis=1)
        metas: Corpus.MetaRegistry = Corpus.MetaRegistry()
        for col in meta_df.columns:
            # create series meta
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please rename the column.")
            metas[col] = SeriesMeta(col, meta_df.loc[:, col])
        corpus = Corpus(df[col_doc], metas, name)
        return corpus

    def __init__(self, text: pd.Series, metas: Union[dict[str, Meta], MetaRegistry] = None, name: str = None):
        self._name = name if name else generate_name(self)

        text.name = self.COL_DOC
        self._df: pd.DataFrame = pd.DataFrame(text, columns=[self.COL_DOC])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self.COL_DOC, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_DOC} column in dataframe."

        self._parent: Optional[Corpus] = None

        # meta data
        self._meta_registry: Corpus.MetaRegistry = Corpus.MetaRegistry(metas)

        # document term matrix - DTM
        self._dtm_registry: Corpus.DTMRegistry = Corpus.DTMRegistry()

        # processing
        self._processing_history = list()

        # regex patterns
        self._pattern_words = re.compile(r'\w+')
        self._pattern_hashtags = re.compile(r'#[A-Za-z0-9_-]+')
        self._pattern_mentions = re.compile(r'@[A-Za-z0-9_-]+')

        # standard viz
        self._viz = CorpusViz(self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        global _ALL_CORPUS_NAMES
        if name in _ALL_CORPUS_NAMES:
            new_name = name + '_'
            logger.info(f"{name} already exists. It renamed to {new_name}")
            name = new_name
            _ALL_CORPUS_NAMES = _ALL_CORPUS_NAMES.union(name)
        self._name = name

    @property
    def parent(self) -> 'Corpus':
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._parent is None

    # slicing
    @property
    def slicer(self) -> CorpusSlicer:
        return CorpusSlicer(self)

    # document term matrix
    @property
    def dtm(self) -> DTM:
        """ Document-Term Matrix. """
        if not self._dtm_registry.get_tokens_dtm().is_built:
            root = self.find_root()
            root._dtm_registry.get_tokens_dtm().initialise(root.docs())
        return self._dtm_registry.get_tokens_dtm()

    @property
    def custom_dtm(self) -> DTM:
        return self._dtm_registry.get_custom_dtm()

    @property
    def viz(self) -> CorpusViz:
        return self._viz

    def find_root(self) -> 'Corpus':
        """ Find and return the root corpus. """
        if self.is_root: return self
        parent = self._parent
        while not parent.is_root:
            parent = parent._parent
        return parent

    def create_custom_dtm(self, tokeniser_func: Callable[[TDoc], list[str]]) -> DTM:
        """ Detaches from root corpus and then build a custom dtm. """
        _ = self.detached()
        return self._update_custom_dtm(tokeniser_func)

    def _update_custom_dtm(self, tokeniser_func: Callable[[TDoc], list[str]]) -> DTM:
        """ Create a custom DTM based on custom tokeniser function. """
        root = self.find_root()
        dtm = DTM()
        dtm.initialise(root.docs(),
                       vectorizer=CountVectorizer(preprocessor=lambda x: x, tokenizer=tokeniser_func))

        root._dtm_registry.set_custom_dtm(dtm)
        if not self.is_root:
            self._dtm_registry.set_custom_dtm(dtm.cloned(self.docs().index))
        return self._dtm_registry.get_custom_dtm()

    # meta data
    @property
    def meta(self) -> MetaRegistry:
        return self._meta_registry

    # statistics
    @property
    def num_terms(self) -> int:
        return self.dtm.total

    @property
    def vocab(self) -> set[str]:
        return set(self.dtm.vocab(nonzero=True))

    def docs(self) -> 'pd.Series':
        return self._df.loc[:, self.COL_DOC]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        describe_cols_to_drop = ['count', 'std', '25%', '50%', '75%']
        docs_info = pd.Series(self.dtm.docs_size_vector).describe().drop(describe_cols_to_drop).astype(
            int)  # Show only integer numbers.
        # docs_info = docs_info.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        mapper = {row_idx: f"{row_idx} Words per Document" for row_idx in docs_info.index}
        docs_info.rename(index=mapper, inplace=True)

        other_info = pd.Series({
            "Corpus Type": self.__class__.__name__,
            "Number of Documents": len(self),
            "Number of Total Words": self.dtm.total,
            "Size of Vocabulary": len(self.dtm.vocab(nonzero=True)),
        })

        meta_info = pd.Series({
            "metas": ', '.join(self._meta_registry.keys())
        })
        return pd.concat([other_info, docs_info, meta_info]).to_frame(name='')

    def sample(self, n: int, rand_stat=None) -> 'Corpus':
        """ Uniformly sample from the corpus. """
        mask = self._df.isna().squeeze()  # Return a mask of all False
        mask[mask.sample(n=n, random_state=rand_stat).index] = True
        return self.cloned(mask)

    def add_meta(self, meta: Meta):
        if meta.id in self._meta_registry.keys(): raise ValueError(f"{meta.id} already exists.")
        if isinstance(meta, SeriesMeta) and not meta.series.index.equals(self._df.index):
            meta.series.set_axis(self._df.index, inplace=True)
        self._meta_registry[meta.id] = meta

    def remove_meta(self, id_: str):
        del self._meta_registry[id_]

    def update_meta(self, meta: Meta):
        if isinstance(meta, SeriesMeta) and not meta.series.index.equals(self._df.index):
            meta.series.set_axis(self._df.index, inplace=True)
        self._meta_registry[meta.id] = meta

    def generate_words(self) -> Generator[str, None, None]:
        """ Generate list of words for each document in the corpus. """
        texts = self.docs()
        for i in range(len(texts)):
            for word in self._gen_words_from(texts.iloc[i]):
                yield word

    def _gen_words_from(self, doc) -> Generator[str, None, None]:
        return (token.lower() for token in self._pattern_words.findall(doc))

    def generate_hashtags(self) -> Generator[str, None, None]:
        for doc in self.docs():
            for word in self._gen_hashtags_from(doc):
                yield word

    def _gen_hashtags_from(self, doc: str):
        return (ht for ht in self._pattern_hashtags.findall(doc))

    def generate_mentions(self) -> Generator[str, None, None]:
        for doc in self.docs():
            for word in self._gen_mentions_from(doc):
                yield word

    def _gen_mentions_from(self, doc: str):
        return (m for m in self._pattern_mentions.findall(doc))

    def cloned(self, mask: 'pd.Series[bool]') -> 'Corpus':
        """ Returns a (usually smaller) clone of itself with the boolean mask applied. """
        cloned_docs = self._cloned_docs(mask)
        cloned_metas = self._cloned_metas(mask)
        cloned_dtms = self._cloned_dtms(mask)

        clone = Corpus(cloned_docs, cloned_metas)
        clone._dtm_registry = cloned_dtms
        clone._parent = self
        return clone

    def _cloned_docs(self, mask) -> pd.Series:
        return self.docs().loc[mask]

    def _cloned_metas(self, mask) -> MetaRegistry:
        cloned_meta_registry = Corpus.MetaRegistry()
        for id_, meta in self._meta_registry.items():
            cloned_meta_registry[id_] = meta.cloned(texts=self._df.loc[:, self.COL_DOC], mask=mask)
        return cloned_meta_registry

    def _cloned_dtms(self, mask) -> DTMRegistry:
        registry = Corpus.DTMRegistry()
        registry.set_tokens_dtm(self._dtm_registry.get_tokens_dtm().cloned(mask))
        if self._dtm_registry.get_custom_dtm() is not None:
            registry.set_custom_dtm(self._dtm_registry.get_custom_dtm().cloned(mask))
        return registry

    def detached(self) -> 'Corpus':
        """ Detaches from corpus tree and becomes the root.

        DTM will be regenerated when accessed - hence a different vocab.
        """
        self._parent = None
        self._dtm_registry = Corpus.DTMRegistry()
        meta_reg = Corpus.MetaRegistry()
        for k, meta in self.meta.items():
            if isinstance(meta, SeriesMeta):
                sm = SeriesMeta(meta.id, meta.series.copy().reset_index(drop=True))
                meta_reg[sm.id] = sm
            else:
                meta_reg[k] = meta
        self._meta_registry = meta_reg
        self._df = self._df.copy().reset_index(drop=True)
        return self

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc('text')
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]

    def __getitem__(self, item):
        if isinstance(item, int):
            mask = self._df.index == self._df.iloc[item].name
        else:  # i.e. type=slice
            start = item.start
            stop = item.stop
            if start is None: start = 0
            if stop is None: stop = len(self._df)
            if item.step is not None: raise NotImplementedError("Slicing with step is currently not implemented.")
            mask = self._df.iloc[start:stop].index
        return self.cloned(mask)


class SpacyCorpus(Corpus):
    """ SpacyCorpus
    This class inherits from the Corpus class with the added and adjusted functions to handle spacy's Doc data
    structure as opposed to string. However, the original string data structure is kept. These may be accessed via
    `.docs()` and `.texts()` respectively.

    Metadata in this class also includes metadata stored in Doc objects. See class meta/DocMeta. Which may again
    be used for slicing the corpus.

    To build a SpacyCorpus, you'll need to `process()` a Corpus object. See class SpacyProcessor. This will run
    the spacy process and update the corpus's meta registry. You'll still need to load spacy's Language object
    which is used in the process.

    ```
    nlp = spacy.blank('en')
    from juxtorpus.corpus.processors import process
    spcycorpus = process(corpus, nlp=nlp)
    ```

    Subtle differences to Corpus:
    As spacy utilises the tokeniser set out by the Language object, you may find summary statistics to be inconsistent
    with the Corpus object you had before it was processed into a SpacyCorpus.
    """

    @classmethod
    def from_corpus(cls, corpus: Corpus, nlp: Language, source=None) -> 'SpacyCorpus':
        from juxtorpus.corpus.processors import process
        return process(corpus, nlp=nlp, source=source, name=corpus.name)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_doc: str = Corpus.COL_DOC,
                       nlp: Language = spacy.blank('en')) -> 'SpacyCorpus':
        corpus = super().from_dataframe(df, col_doc)
        return cls.from_corpus(corpus, nlp)

    def __init__(self, docs, metas: dict, nlp: spacy.Language, source: str, name: str = None):
        super(SpacyCorpus, self).__init__(docs, metas, name)
        self._nlp = nlp
        self._source = source
        self.source_to_word_matcher = {
            None: is_word,
            'tweets': is_word_tweets,
        }
        matcher_func = self.source_to_word_matcher.get(source, None)
        if matcher_func is None:
            raise LookupError(f"Source {source} is not supported. "
                              f"Must be one of {', '.join(self.source_to_word_matcher.keys())}")

        self._is_word_matcher = matcher_func(self._nlp.vocab)
        self._is_hashtag_matcher = is_hashtag(self._nlp.vocab)
        self._is_mention_matcher = is_mention(self._nlp.vocab)

    @property
    def source(self):
        return self._source

    @property
    def nlp(self) -> Language:
        return self._nlp

    @property
    def slicer(self) -> SpacyCorpusSlicer:
        return SpacyCorpusSlicer(self)

    @property
    def dtm(self) -> DTM:
        if not self._dtm_registry.get_tokens_dtm().is_built:
            root = self.find_root()
            root._dtm_registry.get_tokens_dtm().initialise(root.docs(),
                                                           vectorizer=CountVectorizer(preprocessor=lambda x: x,
                                                                                      tokenizer=self._gen_words_from))
        return self._dtm_registry.get_tokens_dtm()

    def cloned(self, mask: 'pd.Series[bool]') -> 'SpacyCorpus':
        clone = super().cloned(mask)
        scorpus = SpacyCorpus(clone.docs(), clone.meta, self.nlp, self._source)
        scorpus._dtm_registry = clone._dtm_registry
        scorpus._parent = self
        return scorpus

    def summary(self, spacy: bool = False):
        df = super(SpacyCorpus, self).summary()
        if spacy:
            spacy_info = {
                'lang': self.nlp.meta.get('lang'),
                'model': self.nlp.meta.get('name'),
                'pipeline': ', '.join(self.nlp.pipe_names)
            }
            return pd.concat([df, pd.DataFrame.from_dict(spacy_info, orient='index')])
        return df

    def _gen_words_from(self, doc: Doc):
        return (doc[start: end].text.lower() for _, start, end in self._is_word_matcher(doc))

    def _gen_hashtags_from(self, doc: Doc):
        return (doc[start: end].text.lower() for _, start, end in self._is_hashtag_matcher(doc))

    def _gen_mentions_from(self, doc: Doc):
        return (doc[start: end].text.lower() for _, start, end in self._is_mention_matcher(doc))

    def generate_lemmas(self) -> Generator[str, None, None]:
        texts = self.docs()
        for i in range(len(texts)):
            yield self._gen_lemmas_from(texts.iloc[i])

    def _gen_lemmas_from(self, doc):
        return (doc[start: end].lemma_ for _, start, end in self._is_word_matcher(doc))
