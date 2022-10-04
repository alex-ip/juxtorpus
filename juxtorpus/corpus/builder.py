import pandas as pd
import pathlib
from functools import partial
from typing import Union, Callable

from juxtorpus.corpus import Corpus
from juxtorpus.meta import SeriesMeta
from juxtorpus.loader import LazySeries


class CorpusBuilder(object):
    def __init__(self, paths: Union[str, pathlib.Path, list[pathlib.Path]]):
        if isinstance(paths, str):
            paths = pathlib.Path(paths)
        if isinstance(paths, pathlib.Path):
            paths = [paths]
        self._paths = paths
        self._corpus_type: str = 'corpus'
        self._nrows = None
        self._metas_configs = dict()
        self._sep = ','
        self._col_text = 'text'
        self._columns = pd.read_csv(self._paths[0], nrows=0).columns

        self._preprocessors = list()

    def head(self, n: int):
        return pd.read_csv(self._paths[0], nrows=n).head(n)

    def show_columns(self):
        all = pd.Series(self._columns, name='All Columns')
        added = pd.Series((col for col in self._metas_configs.keys()), name='Added')
        df = pd.concat([all, added], axis=1).fillna('').T
        return df

    def add_meta(self, column: str, dtype: str = None, lazy=False):
        if column not in self._columns:
            raise ValueError(f"{column} column does not exist.")
        col_dict = self._metas_configs.get(column, dict())
        col_dict['dtype'] = dtype
        col_dict['load_type'] = 'series' if not lazy else 'lazy'
        self._metas_configs[column] = col_dict

    def remove_meta(self, column: str):
        del self._metas_configs[column]

    def set_text_column(self, column: str):
        if column not in self._columns:
            raise KeyError(
                f"Column: '{column}' not found. Use show_columns() to preview the columns in the dataframe")
        self._col_text = column

    def set_sep(self, sep: str):
        self._sep = sep

    def set_nrows(self, nrows: int):
        if nrows < 0:
            raise ValueError("nrows must be a positive integer. Set as None to remove.")
        self._nrows = nrows

    def set_corpus_type(self, type_: str):
        self._corpus_type = type_

    def set_preprocessors(self, preprocess_callables: list[Callable]):
        """ Set a list of preprocessors for your text data.

        This is typically designed for basic preprocessing.
        Your preprocessor callables/functions will have the text passed down.
        """
        self._preprocessors = preprocess_callables

    def _preprocess(self, text):
        for preprocessor in self._preprocessors:
            text = preprocessor(text)
        return text

    def build(self) -> 'Corpus':
        # decide which corpus.
        corpus_cls = Corpus
        # if self._corpus_type.upper() == 'TWEET':
        #     corpus_cls = TweetCorpus

        metas = dict()
        lazies = {col: col_dict for col, col_dict in self._metas_configs.items()
                  if col_dict.get('load_type') == 'lazy'}

        # note: usecols is actually buggy if used with lambda func. - see commit: 5369ad6
        for col, col_dict in lazies.items():
            read_func = partial(pd.read_csv, usecols=[col], sep=self._sep)
            lazy_series = LazySeries(self._paths, self._nrows, col_dict.get('dtype'), read_func)
            meta = SeriesMeta(col, lazy_series)
            metas[col] = meta

        series_cols = set(
            (col for col, col_dict in self._metas_configs.items() if col_dict.get('load_type') == 'series')
        )
        datetime_cols = set((col for col in series_cols if self._metas_configs.get(col).get('dtype') == 'datetime'))

        series_cols = series_cols.union([self._col_text])
        current = 0
        dfs = list()
        for path in self._paths:
            if self._nrows is None:
                df = pd.read_csv(path, nrows=self._nrows, usecols=series_cols, sep=self._sep,
                                 parse_dates=list(datetime_cols))
            else:
                if current >= self._nrows:
                    break
                df = pd.read_csv(path, nrows=self._nrows - current, usecols=series_cols, sep=self._sep,
                                 parse_dates=list(datetime_cols))
                current += len(df)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)

        # dtypes conversions
        dtype_dict = {col: self._metas_configs.get(col).get('dtype')
                      for col in series_cols.difference([self._col_text]).difference(datetime_cols)}
        dtype_dict[self._col_text] = pd.StringDtype(storage='pyarrow')
        df = df.astype(dtype_dict)

        if self._col_text not in df.columns:
            raise KeyError(f"{self._col_text} column is missing. This column is compulsory.")
        # 3. apply preprocessors if exist
        df[self._col_text] = df.loc[:, self._col_text].apply(self._preprocess)

        # 4. set up Corpus dependencies
        series_text = df.loc[:, self._col_text]
        # 5. update the rest of the meta dictionary - build metas
        series_cols.remove(self._col_text)
        for col in series_cols:
            series = df[col]
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please use a different column name.")
            metas[col] = SeriesMeta(col, series)
        return corpus_cls(series_text, metas=metas)


if __name__ == '__main__':
    cb = CorpusBuilder(
        # paths=['/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv']
        paths=[pathlib.Path('~/Downloads/elonmusk_tweets.csv')]
    )
    cb.set_nrows(100)
    cb.set_corpus_type('corpus')
    cb.set_sep(',')
    # cb.set_text_column('processed_text')
    # for col in ['year', 'day', 'tweet_lga', 'lga_code_2020']:
    #     cb.add_meta(col, lazy=True)
    # cb.add_meta('month', dtype='string', lazy=False)

    cb.set_text_column('doc')
    cb.add_meta('created_at', dtype='datetime', lazy=True)

    print(cb.show_columns())
    corpus = cb.build()
    print(corpus.metas())

    # print(corpus.get_meta('tweet_lga').preview(5))
    print(corpus.get_meta('created_at').head(5))
