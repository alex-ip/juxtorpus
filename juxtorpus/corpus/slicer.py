from abc import ABC

import pandas as pd

from juxtorpus.corpus import Corpus, SpacyCorpus
from juxtorpus.corpus.meta import *

from typing import Union, Callable, Optional
from spacy.matcher import Matcher
from spacy.tokens import Doc
import weakref
import re

import colorlog

logger = colorlog.getLogger(__name__)


def slicer(corpus):
    if isinstance(corpus, SpacyCorpus):
        return SpacyCorpusSlicer(corpus)
    if isinstance(corpus, Corpus):
        return CorpusSlicer(corpus)
    raise ValueError(f"corpus must be an instance of {Corpus.__name__}. Got {type(corpus)}.")


class CorpusSlice(Corpus):
    def __init__(self, parent_corpus: weakref.ReferenceType[Corpus], *args, **kwargs):
        super(CorpusSlice, self).__init__(*args, **kwargs)


class CorpusSlices(dict):
    def join(self):
        pass  # do alignment of dict if no original corpus?


class FrozenCorpusSlices(CorpusSlices):
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
        raise RuntimeError("You may not write to FrozenCorpusSlices.")


class CorpusSlicer(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def filter_by_condition(self, id_, cond_func: Callable[[Any], bool]):
        meta = self._get_meta_or_raise_err(id_)

        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def filter_by_item(self, id_, items):
        meta = self._get_meta_or_raise_err(id_)
        cond_func = self._item_cond_func(items)
        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def _item_cond_func(self, items):
        items = [items] if isinstance(items, str) else items
        items = set(items)

        def cond_func(any_):
            if isinstance(any_, str):
                return any_ in items
            elif isinstance(any_, dict):
                return not set(any_.keys()).isdisjoint(items)
            elif isinstance(any_, Iterable):
                return not set(any_).isdisjoint(items)
            else:
                raise TypeError(f"Unable to filter {type(any_)}. Only string or iterables.")

        return cond_func

    def filter_by_regex(self, id_, regex: str, ignore_case: bool):
        meta = self._get_meta_or_raise_err(id_)
        flags = 0 if not ignore_case else re.IGNORECASE
        pattern = re.compile(regex, flags=flags)

        cond_func = lambda any_: pattern.search(any_) is not None
        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def filter_by_datetime(self, id_, after: Optional[str] = None, before: Optional[str] = None):
        """ Filter by datetime
        :arg after - any datetime string recognised by pandas.
        :arg before - any datetime string recognised by pandas.
        You must provide either 'after' or 'before' or both.
        """
        meta = self._get_meta_or_raise_err(id_)
        if after is None and before is None: raise ValueError("You must either set 'after' or 'before' or both.")
        if after is not None: after = pd.to_datetime(after)
        if before is not None: before = pd.to_datetime(before)
        if after is not None and before is not None:
            cond_func = lambda dt: after < dt < before
        elif after is not None and before is None:
            cond_func = lambda dt: after < dt
        else:
            cond_func = lambda dt: dt < before
        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def _mask_by_condition(self, meta, cond_func):
        mask = meta.apply(cond_func)
        try:
            mask = mask.astype('boolean')
        except TypeError:
            raise TypeError("Does your condition function return booleans?")
        return mask

    def group_by_conditions(self, id_, cond_funcs: list[Callable]):
        """ TODO: basically runs filter by condition multiple times and organise into FrozenCorpusSlices. """
        raise NotImplementedError()

    def group_by(self, id_, grouper: pd.Grouper = None):
        """ Return groups of the subcorpus based on their metadata.

        :arg grouper: pd.Grouper - as you would in pandas
        :return tuple[groupid, subcorpus]
        """
        meta = self._get_meta_or_raise_err(id_)
        if not isinstance(meta, SeriesMeta):
            raise NotImplementedError(f"Unable to groupby non SeriesMeta. "
                                      f"Please use {self.group_by_conditions.__name__}.")
        series = meta.series()
        # using pd.Grouper on datetime requires it to be an index.
        if grouper is not None:
            by = grouper
            if pd.api.types.is_datetime64_any_dtype(series):
                by.level = series.name
                df = self.corpus._df.set_index([self.corpus._df.index, series])
                return ((gid, self.corpus.cloned(g.index.droplevel(by.level)))
                        for gid, g in df.groupby(by, axis=0, group_keys=True))
        else:
            by = series
        return ((gid, self.corpus.cloned(g.index)) for gid, g in series.groupby(by=by, axis=0, group_keys=True))

    def _get_meta_or_raise_err(self, id_):
        meta = self.corpus.meta.get(id_)
        if meta is None: raise KeyError(f"{id_} metadata does not exist. Try calling metas().")
        return meta


class SpacyCorpusSlicer(CorpusSlicer, ABC):
    def __init__(self, corpus: SpacyCorpus):
        if not isinstance(corpus, SpacyCorpus): raise ValueError(f"Must be a SpacyCorpus. Got {type(corpus)}.")
        super(SpacyCorpusSlicer, self).__init__(corpus)

    def filter_by_matcher(self, matcher: Matcher):
        cond_func = self._matcher_cond_func(matcher)
        docs = self.corpus.docs()
        if isinstance(docs, pd.Series):
            mask = docs.apply(cond_func).astype('boolean')
        else:
            mask = map(cond_func, docs)  # inf corpus. Corpus class itself does not support this yet. placeholder.
        return self.corpus.cloned(mask)

    def _matcher_cond_func(self, matcher):
        def _cond_func(doc: Doc):
            return len(matcher(doc)) > 0

        return _cond_func


if __name__ == '__main__':
    metas: dict[str, Meta] = {
        'col': SeriesMeta('col', pd.Series(['z', 'y', 'x'])),
        'col_num': SeriesMeta('col_2', pd.Series([1, 2, 3]))
    }
    corp = Corpus(pd.Series(['a', 'b', 'c']), metas)

    slicer = CorpusSlicer(corp)
    subset = slicer.filter_by_condition('col', lambda x: x == 'z')
    print(subset._df)
    subset = slicer.filter_by_item('col', {'y', 'x'})
    print(subset._df)
