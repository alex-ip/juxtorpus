from juxtorpus.corpus import Corpus
from juxtorpus.meta import *
from typing import Union, List, Callable
import weakref
import re


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
            elif isinstance(any_, Iterable):
                return not set(any_).isdisjoint(items)
            else:
                print(f"[Warn] Unable to filter {type(any_)}. Only string or iterables.")

        return cond_func

    def filter_by_regex(self, id_, regex: str, ignore_case: bool):
        meta = self._get_meta_or_raise_err(id_)
        flags = 0 if not ignore_case else re.IGNORECASE
        pattern = re.compile(regex, flags=flags)
        cond_func = lambda x: pattern.search(x) is not None

        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def _mask_by_condition(self, meta, cond_func):
        mask = meta.apply(cond_func)
        try:
            mask = mask.astype('boolean')
        except TypeError:
            raise TypeError("Does your function return a boolean?")
        return mask

    def slice_by_condition(self, id_, cond_func):
        """ TODO: basically runs filter by condition multiple times and organise into FrozenCorpusSlices. """
        raise NotImplementedError()

    def group_by(self, id_):
        """ TODO: retrieve the series and run pandas groupby. Only works for type: SeriesMeta. """
        raise NotImplementedError()

    def _get_meta_or_raise_err(self, id_):
        meta = self.corpus.get_meta(id_)
        if meta is None: raise KeyError(f"{id_} metadata does not exist. Try calling metas().")
        return meta


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
