from abc import ABC

from typing import Union, Callable, Optional, Any, Generator
from spacy.matcher import Matcher
from spacy.tokens import Doc
import re

from juxtorpus.corpus.meta import *
from juxtorpus.viz import Widget
from juxtorpus.viz.widgets.corpus.slicer import SlicerWidget
from juxtorpus.corpus.operation import *
import colorlog

logger = colorlog.getLogger(__name__)


class CorpusSlicer(Widget):
    """ CorpusSlicer

    The corpus slicer is used in conjunction with the Corpus class to serve its main design feature:
    the ability to recursively slice and dice the corpus.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def filter_by_condition(self, id_, cond_func: Callable[[Any], bool]):
        """ Filter by condition
        :arg id_ - meta's id
        :arg cond_func -  Callable that returns a boolean.
        """
        meta = self.corpus.meta.get_or_raise_err(id_)

        mask = self._mask_by_condition(meta, cond_func)
        return self.corpus.cloned(mask)

    def filter_by_item(self, id_, items):
        """ Filter by item - items can be str or numeric types.

        :arg id_ - meta's id.
        :arg items - the list of items to include OR just a single item.
        """
        meta = self.corpus.meta.get_or_raise_err(id_)
        op = ItemOp(meta, items)
        return self.corpus.cloned(op.mask())

    def filter_by_range(self, id_, min_: Optional[Union[int, float]] = None, max_: Optional[Union[int, float]] = None):
        """ Filter by a range [min, max). Max is non inclusive. """
        meta = self.corpus.meta.get_or_raise_err(id_)
        if min_ is None and max_ is None: return self.corpus
        op = RangeOp(meta, min_, max_)
        return self.corpus.cloned(op.mask())

    def filter_by_regex(self, id_, regex: str, ignore_case: bool = False):
        """ Filter by regex.
        :arg id - meta id
        :arg regex - the regex pattern
        :arg ignore_case - whether to ignore case
        """
        meta = self.corpus.meta.get_or_raise_err(id_)
        op = RegexOp(meta, regex, ignore_case)
        return self.corpus.cloned(op.mask())

    def filter_by_datetime(self, id_, start: Optional[str] = None, end: Optional[str] = None,
                           strftime: Optional[str] = None):
        """ Filter by datetime in range (start, end].
        :arg start - any datetime string recognised by pandas.
        :arg end - any datetime string recognised by pandas.
        :arg strftime - datetime string format

        If no start or end is provided, it'll return the corpus unsliced.
        """
        meta = self.corpus.meta.get_or_raise_err(id_)
        if start is None and end is None: return self.corpus
        op = DatetimeOp(meta, start, end, strftime)
        return self.corpus.cloned(op.mask())

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

    def group_by(self, id_, grouper: pd.Grouper = None) -> Generator[tuple[str, 'Corpus'], None, None]:
        """ Return groups of the subcorpus based on their metadata.

        :arg grouper: pd.Grouper - as you would in pandas
        :return tuple[groupid, subcorpus]
        """
        meta = self.corpus.meta.get_or_raise_err(id_)
        if not isinstance(meta, SeriesMeta):
            raise NotImplementedError(f"Unable to groupby non SeriesMeta. "
                                      f"Please use {self.group_by_conditions.__name__}.")
        return ((gid, self.corpus.cloned(mask)) for gid, mask in meta.groupby(grouper))

    def widget(self):
        return SlicerWidget(self.corpus).widget()


class SpacyCorpusSlicer(CorpusSlicer, ABC):
    def __init__(self, corpus: 'SpacyCorpus'):
        from juxtorpus.corpus import SpacyCorpus
        if not isinstance(corpus, SpacyCorpus): raise ValueError(f"Must be a SpacyCorpus. Got {type(corpus)}.")
        super(SpacyCorpusSlicer, self).__init__(corpus)

    def filter_by_matcher(self, matcher: Matcher):
        """ Filter by matcher
        If the matcher matches anything, that document is kept in the sliced corpus.
        """
        op = MatcherOp(self.corpus.docs(), matcher)
        return self.corpus.cloned(op.mask(self.corpus.docs()))
