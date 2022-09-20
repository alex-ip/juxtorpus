from abc import ABCMeta, abstractmethod
from typing import List, Callable, Any, Set, Union, Iterable
import pandas as pd
from spacy.tokens import Doc
import pathlib
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
    def cloned(self, mask):
        raise NotImplementedError()

    @abstractmethod
    def preview(self, n: int):
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

    def cloned(self, mask):
        return SeriesMeta(self._id, self.series()[mask])

    def preview(self, n):
        return self.series().head(n)


class DelimitedStrSeriesMeta(SeriesMeta):
    def __init__(self, id_, series: pd.Series, delimiter: str):
        super(DelimitedStrSeriesMeta, self).__init__(id_, series)
        self.delimiter = delimiter

    def apply(self, func):
        return self.series.apply(lambda x: x.split(self.delimiter)).apply(func)

    def cloned(self, mask):
        return DelimitedStrSeriesMeta(self._id, self.series[mask], self.delimiter)


""" Metadata from spaCy docs can only be derived metadata. """


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id_: str, attr: str, doc_generator: Callable[[], Iterable[Doc]]):
        super(DocMeta, self).__init__(id_)
        self._attr = attr
        self._doc_generator = doc_generator

    @property
    def attribute(self):
        return self._attr

    def apply(self, func) -> pd.Series:
        pass

    def cloned(self, mask):
        doc_generator = mask
        return DocMeta(self._id, self._attr, doc_generator)

    def preview(self, n: int):
        return [doc for i, doc in enumerate(self._doc_generator()) if i < n]

    def __iter__(self):
        for doc in self._doc_generator():
            yield doc

    def _get_doc_attr(self, doc: Doc) -> Any:
        """ Returns a built-in spacy entity OR a custom entity. """
        return doc.get_extension(self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)

    def __repr__(self):
        return f"{super(DocMeta, self).__repr__()[:-2]}, Attribute: {self._attr}]"
