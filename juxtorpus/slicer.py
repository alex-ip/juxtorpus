from juxtorpus.corpus import Corpus
from typing import Union, List, Callable
import weakref


class CorpusSlicer(object):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def filter_by_items(self):
        """ Returns a subset of the corpus that 'have' these items. """
        pass

    def filter_by_match(self):
        pass

    def filter_by_condition(self, meta_id: str, conds: List[Callable]):
        """ Returns a subset of the corpus that satisfies this condition. """
        pass

    def slice_by_item(self, meta_id: str):
        """ Returns a group of corpus slices from unique values of a category. """
        pass

    def slice_by_conditions(self, meta_id: str, conds: List[Callable]):
        """ Returns a group of corpus slices that matches a set of conditions.
        Note: If it matches > 1 condition, it'll be all the groups. (There is data redundancy here)
        """
        pass


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


if __name__ == '__main__':
    user_stories = {
        'filter_by_item': [
            'As an explorer/analyst, I want to find all the tweets with a particular hashtag because I only want to do analysis on a subset of the corpus',
            'As an explorer/analyst, I want to find all the tweets with 2 hashtags because both hashtags are similar.'
        ],
        'filter_by_comparison': [
            'As an explorer/analyst, I want to find all tweets that have > 2 hashtags.'
        ],
        'filter_by_partial_match': [
            'As an explorer/analyst, I want to find all tweets with a particular'
        ],
        'slice_by_item': [
            'As an explorer/analyst, I want to get a group of all hashtags because I want to explore '
        ],
        'slice_by_conditions': [
            'As an explorer/analyst, I want to get a number of groups based on a set of conditions.'
        ]
    }
