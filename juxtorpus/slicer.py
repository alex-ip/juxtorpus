from juxtorpus.corpus import Corpus
from juxtorpus.meta import *
from typing import Union, List, Callable
import weakref


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
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def metas(self):
        return self.corpus.metas()

    """ Filtering for a subset of the corpus. """

    ## Subsetting data based on matches ##
    def filter_by_items(self, meta_id: str, items: List[str], op: str) -> Corpus:
        """ Returns a subset of the corpus that 'have' these items.

        Supported dtypes: string
        """
        meta: ItemMasker
        meta = self.corpus.get_meta(meta_id)
        if meta is None:
            raise KeyError(f"{meta_id} does not exist. Please call .metas() for a list of meta data ids.")
        if not isinstance(meta, ItemMasker):
            raise TypeError(f"{meta_id} can't be filtered by item. It is not an ItemMasker.")
        _mask = meta.mask_on_items(items, op)
        return Corpus(self.corpus._df[_mask], [])  # TODO: replace this None with meta

    def filter_by_regex(self, meta_id: str, regex: str) -> Corpus:
        """ Returns a subset of the corpus that matches these regex.

        Supported dtypes: string,
        """
        raise NotImplementedError("To be implemented...")

    def filter_by_condition(self, meta_id: str, cond: Callable[[Any], bool]):
        """ Returns a subset of the corpus that satisfies this condition. """
        meta: Meta
        meta = self.corpus.get_meta(meta_id)
        if meta is None:
            raise KeyError(f"{meta_id} does not exist. Please call .metas() for a list of meta data ids.")
        _mask = meta.mask_on_condition(cond)
        return Corpus(self.corpus._df[_mask], [])

    def filter_by_conditions(self, meta_id: str, conds: List[Callable]):
        """ Returns a subset of the corpus that satisfies these conditions """
        raise NotImplementedError("To be implemented...")

    """ Slicing the corpus into slices/groups. """

    def slice_by_item(self, meta_id: str) -> FrozenCorpusSlices:
        """ Returns a group of corpus slices from unique values of a category. """
        pass

    def slice_by_conditions(self, meta_id: str, conds: List[Callable]) -> FrozenCorpusSlices:
        """ Returns a group of corpus slices that matches a set of conditions.
        Note: If it matches > 1 condition, it'll be all the groups. (There is data redundancy here)
        """
        pass


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

    from juxtorpus.meta import SeriesMeta, LazySeries, CategoricalSeriesMeta
    from juxtorpus.corpus import TweetCorpus

    # Corpus Builder
    df = pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv', nrows=500)
    df_meta_ = pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv', nrows=1,
                       usecols=lambda col: col != 'text')
    metas = list()
    for meta_col in df_meta_.columns:
        # extract metadata from column
        if meta_col == 'tweet_lga':
            meta = CategoricalSeriesMeta(id_=meta_col, series=LazySeries(
                path='~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv', col=meta_col,
                dtype=None, nrows=500))
        else:
            meta = SeriesMeta(id_=meta_col, series=LazySeries(
                path='~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv', col=meta_col,
                dtype=None, nrows=500))
        metas.append(meta)

    tweets = TweetCorpus(df, metas).preprocess()

    slicer = CorpusSlicer(tweets)
    subset = slicer.filter_by_condition('tweet_lga', lambda x: 'A' in (x))
    print()


