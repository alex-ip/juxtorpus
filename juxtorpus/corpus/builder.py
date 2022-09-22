import pandas as pd
import pathlib
from functools import partial

from juxtorpus.corpus import Corpus, TweetCorpus
from juxtorpus.meta import LazySeries, SeriesMeta


class CorpusBuilder(object):
    def __init__(self, paths: list[pathlib.Path]):
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

    def show_columns(self):
        header = pd.read_csv(self._paths[0], nrows=0)
        all = pd.Series(header.columns, name='All Columns')
        added = pd.Series((added for added_cols in self._metas_configs.values() for added in added_cols), name='Added')
        df = pd.concat([all, added], axis=1).fillna('')
        return df

    def add_meta(self, column: str, lazy=False):
        key: str = 'series'
        if lazy:
            key = 'lazy'
        list_ = self._metas_configs.get(key, list())
        list_.append(column)
        self._metas_configs[key] = list_

    def set_text_column(self, column: str):
        self._col_text = column

    def set_sep(self, sep: str):
        self._sep = sep

    def set_nrows(self, nrows: int):
        self._nrows = nrows

    def set_corpus_type(self, type_: str):
        self._corpus_type = type_

    def build(self) -> 'Corpus':
        # decide which corpus.
        corpus_cls = Corpus
        if self._corpus_type.upper() == 'TWEET':
            corpus_cls = TweetCorpus

        # build all meta data
        metas = dict()
        # 1. lazy metas
        for col in self._metas_configs.get('lazy', []):
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please use a different column name.")

            read_func = partial(pd.read_csv, usecols=[col], sep=self._sep)
            lazy_series = LazySeries(self._paths, self._nrows, read_func)
            meta = SeriesMeta(col, lazy_series)
            metas[col] = meta

        # 2. update the rest of the meta dictionary - build df
        cols_set = set(self._metas_configs.get('series', [])).union({self._col_text})
        current = 0
        dfs = list()
        for path in self._paths:
            if self._nrows is None:
                df = pd.read_csv(path, nrows=self._nrows, usecols=cols_set, sep=self._sep)
            else:
                if current >= self._nrows:
                    break
                df = pd.read_csv(path, nrows=self._nrows - current, usecols=cols_set, sep=self._sep)
                current += len(df)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)

        if self._col_text not in df.columns:
            raise KeyError(f"{self._col_text} column is missing. This column is compulsory.")
        series_text = df.loc[:, self._col_text]

        # 3. update the rest of the meta dictionary - build metas
        cols_set.remove(self._col_text)
        for col in cols_set:
            series = df[col]
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please use a different column name.")
            metas[col] = SeriesMeta(col, series)
        return corpus_cls(series_text, metas=metas)


if __name__ == '__main__':
    cb = CorpusBuilder(
        paths=['/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv']
    )
    cb.set_nrows(100)
    cb.set_corpus_type('corpus')
    cb.set_sep(',')
    cb.set_text_column('processed_text')

    corpus = cb.build()
    print(corpus.metas())

    # add some meta data now using the same builder - state is kept from above.
    for col in ['year', 'day', 'tweet_lga', 'lga_code_2020']:
        cb.add_meta(col, lazy=True)
    cb.add_meta('month', lazy=False)
    print(cb.show_columns())

    corpus_with_meta = cb.build()
    print(corpus_with_meta.get_meta('tweet_lga'))
    print("+++++")
    print(corpus_with_meta.get_meta('tweet_lga').preview(5))
