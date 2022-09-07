from abc import ABCMeta, abstractmethod
import pandas as pd
from spacy.tokens import Doc
from typing import Union, List, Callable


class Masker(metaclass=ABCMeta):
    """ Base Slicer class

    Subclasses of this abstract class will take in a series and return a boolean mask
    to be used for extracting rows.
    """

    @abstractmethod
    def filter(self, series: pd.Series, items: Union[str, List[str]], cond: Callable = None):
        """ Filter for items in the series based on condition. """
        raise NotImplementedError()


class DocMasker(Masker):
    def __init__(self, spacy_attr: str):
        self.spacy_attr = spacy_attr

    def filter(self, series: 'pd.Series[Doc]',
               items: Union[str, List[str]],
               cond: Callable[[Doc, str, str], bool] = None) -> 'pd.Series[bool]':
        if cond is None:
            cond = DocMasker.check_doc_contains

        if isinstance(items, str):
            return series.apply(lambda doc: cond(doc, self.spacy_attr, items))
        elif isinstance(items, list):
            # NOTE: this is via AND relation. i.e. all items MUST co-exist.
            init_series = pd.Series([False for i in range(len(series))])
            for i, item in enumerate(items):
                if i == 0:
                    init_series = init_series | series.apply(lambda doc: cond(doc, self.spacy_attr, item))
                else:
                    init_series = init_series & series.apply(lambda doc: cond(doc, self.spacy_attr, item))
            return init_series
        else:
            raise ValueError("items must be either a string or a list of strings")

    @classmethod
    def check_doc_contains(cls, doc: Doc, attr: str, item: str) -> bool:
        """ Masking function """
        # custom extensions
        if doc.has_extension(attr):
            return item in (attr_item.text for attr_item in doc.get_extension(attr))
        # spacy built in attributes
        attr_items = getattr(doc, attr, None)
        if attr_items is None:
            raise KeyError(f"'{attr}' attribute or extension does not exist in spacy doc.")
        return item in (attr_item.text for attr_item in attr_items)


if __name__ == '__main__':
    from juxtorpus import nlp

    series = pd.Series(["I am at New York City!",
                        "hello there",
                        "Australia is in the southern hemisphere",
                        "Sydney is 16,200 km from New York City"
                        ])
    items = ['New York City', 'Sydney']
    print(f"Filter for rows where all {items} exists...")
    x = DocMasker(spacy_attr='ents').filter(series.apply(lambda x: nlp(x)), ['New York City', 'Sydney'])
    print(pd.concat([series, x], axis=1))
    print("++++masked++++")
    print(series[x])
    print(f"{len(series[x])} rows found.")
