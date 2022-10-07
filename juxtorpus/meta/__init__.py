import pandas as pd
from abc import ABCMeta, abstractmethod
from spacy import Language
from spacy.tokens import Doc
from typing import List, Callable, Any, Set, Union, Iterable, Generator
from functools import partial

from juxtorpus.loader import LazySeries

""" 
A Collection of data classes representing Corpus Metadata.
"""


class Meta(metaclass=ABCMeta):
    def __init__(self, id_: str):
        self._id = id_

    @property
    def id(self):
        return self._id

    @abstractmethod
    def apply(self, func) -> pd.Series:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def cloned(self, texts: 'pd.Series[str]', mask: 'pd.Series[bool]'):
        raise NotImplementedError()

    @abstractmethod
    def head(self, n: int):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [Id: {self.id}]>"


class SeriesMeta(Meta):

    def __init__(self, id_, series: Union[pd.Series, LazySeries]):
        super(SeriesMeta, self).__init__(id_)
        self._series = series
        # print(self._series)

    def series(self):
        if isinstance(self._series, LazySeries):
            self._series = self._series.load()
        return self._series

    def apply(self, func):
        return self.series().apply(func)

    def __iter__(self):
        for x in iter(self.series.__iter__()):
            yield x

    def cloned(self, texts, mask):
        return SeriesMeta(self._id, self.series()[mask])

    def head(self, n):
        return self.series().head(n)


class DelimitedStrSeriesMeta(SeriesMeta):
    def __init__(self, id_, series: pd.Series, delimiter: str):
        super(DelimitedStrSeriesMeta, self).__init__(id_, series)
        self.delimiter = delimiter

    def apply(self, func):
        return self.series().apply(lambda x: x.split(self.delimiter)).apply(func)

    def cloned(self, texts, mask):
        return DelimitedStrSeriesMeta(self._id, self.series[mask], self.delimiter)


""" Metadata from spaCy docs can only be derived metadata. """


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id_: str, attr: str,
                 nlp: Language, docs: Union[pd.Series, Callable[[], Iterable[Doc]]]):
        super(DocMeta, self).__init__(id_)
        self._attr = attr
        self._docs = docs
        self._nlp = nlp  # keep a ref to the spacy.Language

    @property
    def attr(self):
        return self._attr

    def apply(self, func) -> pd.Series:
        def _inner_func_on_attr(doc: Doc):
            return func(self._get_doc_attr(doc))

        if isinstance(self._docs, pd.Series):
            return self._docs.apply(_inner_func_on_attr)
        return pd.Series(map(_inner_func_on_attr, self._docs()))  # faster than loop. But can be improved.

    def cloned(self, texts, mask):
        # use the series mask to clone itself.
        if isinstance(self._docs, pd.Series):
            return DocMeta(self._id, self._attr, self._nlp, self._docs[mask])
        return DocMeta(self._id, self._attr, self._nlp, partial(self._nlp.pipe, texts))

    def head(self, n: int):
        docs = self._get_iterable()
        texts = (doc.text for i, doc in enumerate(docs) if i < n)
        attrs = (self._get_doc_attr(doc) for i, doc in enumerate(docs) if i < n)
        return pd.DataFrame(zip(texts, attrs), columns=['text', self._id])

    def __iter__(self):
        for doc in self._get_iterable():
            yield doc

    def _get_iterable(self):
        docs: Iterable
        if isinstance(self._docs, pd.Series):
            docs = self._docs
        elif isinstance(self._docs, Callable):
            docs = self._docs()
        else:
            raise RuntimeError(f"docs are neither a Series or a Callable stream. This should not happen.")
        return docs

    def _get_doc_attr(self, doc: Doc) -> Any:
        """ Returns a built-in spacy entity OR a custom entity. """
        # return doc.get_extension(self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)
        return getattr(getattr(doc, '_'), self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)

    def __repr__(self):
        return f"{super(DocMeta, self).__repr__()[:-2]}, Attribute: {self._attr}]"
