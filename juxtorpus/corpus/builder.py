import pandas as pd
import pathlib
from functools import partial
from typing import Union, Callable, Iterable
from dataclasses import dataclass

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.meta import SeriesMeta
from juxtorpus.loader import LazySeries

import colorlog

logger = colorlog.getLogger(__name__)


class MetaConfig(object):
    def __init__(self, column: str, dtype: str, lazy: bool):
        self._column = column
        self._lazy = lazy
        self._dtype = dtype

    @property
    def column(self):
        return self._column

    @property
    def lazy(self):
        return self._lazy

    @property
    def dtype(self):
        return self._dtype

    def __hash__(self):
        # used in checks in case of duplicate column names.
        return hash((self.column, self.__class__.__name__))

    def __eq__(self, other: Union['MetaConfig', str]):
        if not isinstance(other, MetaConfig) and not isinstance(other, str):
            return NotImplemented
        if isinstance(other, str):
            other = (other, self.__class__.__name__)
        return hash(other) == self.__hash__()

    def __repr__(self):
        return f"<{self.column}: dtype={self._dtype} lazy={self._lazy}>"


class DateTimeMetaConfig(MetaConfig):
    COL_DATETIME = 'datetime'

    def __init__(self, columns: Union[str, list[str]], lazy: bool):
        if isinstance(columns, str):
            column = columns
            self.columns = None
        else:
            column = self.COL_DATETIME
            self.columns = columns
        super(DateTimeMetaConfig, self).__init__(column, dtype='datetime', lazy=lazy)

    def is_multi_columned(self) -> bool:
        return self.columns is not None

    def get_parsed_dates(self):
        if self.is_multi_columned():
            return {self.column: self.columns}
        else:
            return [self.column]


class CorpusBuilder(object):
    def __init__(self, paths: Union[str, pathlib.Path, list[pathlib.Path]]):
        if isinstance(paths, str):
            paths = pathlib.Path(paths)
        if isinstance(paths, pathlib.Path):
            paths = [paths]
        self._paths = paths
        self._nrows = None
        self._meta_configs = dict()
        self._sep = ','
        self._col_text = None
        self._columns = pd.read_csv(self._paths[0], nrows=0).columns

        self._preprocessors = list()

    def head(self, n: int):
        return pd.read_csv(self._paths[0], nrows=n).head(n)

    def show_columns(self):
        all = pd.Series(self._columns, name='All Columns')
        df = pd.DataFrame(index=all)
        to_add = list()
        for mc in self._meta_configs.values():
            if type(mc) == DateTimeMetaConfig and mc.is_multi_columned():
                to_add.extend(mc.columns)
            else:
                to_add.append(mc.column)
        df['Add'] = df.index.isin(to_add)
        return df.sort_values(by='Add', ascending=False)

    def add_metas(self, columns: Union[str, list[str]],
                  dtypes: Union[None, str, list[str]] = None,
                  lazy=True):
        """ Add a column to add as metadata OR a list of columns to add.

        :param columns: The columns to add to the corpus.
        :param dtypes: The dtypes to specify.
        :param lazy: Keep series on disk until required.(Default: True)

        If dtype is None, dtype is inferred by pandas.
        """
        if dtypes == 'datetime':
            self._add_datetime_meta(columns, lazy)
            return
        # non datetime columns
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(dtypes, list) and len(columns) != len(dtypes):
            raise ValueError("Number of columns and dtypes must match.")
        else:
            for i in range(len(columns)):
                dtype = dtypes[i] if isinstance(dtypes, list) else dtypes
                self._add_meta(columns[i], dtype, lazy)

    def _add_meta(self, column: str, dtype: str, lazy: bool):
        if column not in self._columns:
            raise ValueError(f"{column} column does not exist.")
        meta_config = MetaConfig(column=column, dtype=dtype, lazy=lazy)
        self._meta_configs[meta_config.column] = meta_config

    def _add_datetime_meta(self, columns: Union[str, list[str]], lazy: bool):
        if isinstance(columns, list):
            for column in columns:
                if column not in self._columns:
                    raise ValueError(f"{column} column does not exist.")
            logger.info(f"You are using a multi-columned datetime. "
                        f"These columns will combined into a single '{DateTimeMetaConfig.COL_DATETIME}' meta.")
        dt_meta_config = DateTimeMetaConfig(columns=columns, lazy=lazy)
        self._meta_configs[dt_meta_config.column] = dt_meta_config

    def remove_metas(self, columns: Union[str, list[str]]):
        # not sure why membership test isn't working with just string.
        # https://docs.python.org/3.9/reference/expressions.html#membership-test-details
        # according to the doc: any(x is e or x == e for e in y) is the underlying implementation.
        # but `column in self._meta_configs` returns false while
        # `any(column is e or column == e for e in self._meta_configs)` returns true.
        # python version 3.9.13
        # self._meta_configs.remove(MetaConfig(column, None, None))  -- decided not to use sets.
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if col in self._meta_configs.keys():
                del self._meta_configs[col]
            else:
                dtmc: DateTimeMetaConfig
                for dtmc in (mc for mc in self._meta_configs.values() if type(mc) == DateTimeMetaConfig):
                    if dtmc.is_multi_columned() and col in dtmc.columns:
                        del self._meta_configs[dtmc.column]

    def update_metas(self, columns: Union[list[str], str],
                     dtypes: Union[None, str, list[str]],
                     lazy: bool):
        self.remove_metas(columns)
        self.add_metas(columns, dtypes, lazy)

    def set_text_column(self, column: str):
        if column not in self._columns:
            raise KeyError(
                f"Column: '{column}' not found. Use show_columns() to preview the columns in the dataframe")
        self._col_text = column

    def set_sep(self, sep: str):
        """ Set the separator to use in parsing the file.
        e.g.
            set_sep(',') for csv            (default)
            set_sep('\t') for tsv
        """
        self._sep = sep

    def set_nrows(self, nrows: int):
        """ Set the number of rows to load into the corpus."""
        if nrows < 0:
            raise ValueError("nrows must be a positive integer. Set as None to remove.")
        self._nrows = nrows

    def set_text_preprocessors(self, preprocess_callables: list[Callable]):
        """ Set a list of preprocessors for your text data.

        This is typically designed for basic preprocessing.
        Your preprocessor callables/functions will have the text passed down.
        """
        if isinstance(preprocess_callables, Callable):
            preprocess_callables = [preprocess_callables]
        self._preprocessors = preprocess_callables

    def _preprocess(self, text):
        for preprocessor in self._preprocessors:
            text = preprocessor(text)
        return text

    def build(self) -> Corpus:
        if self._col_text is None:
            raise ValueError(f"You must set the text column. Try calling {self.set_text_column.__name__} first.")
        metas = dict()
        metas = self._build_lazy_metas(metas)
        metas, texts = self._build_series_meta_and_text(metas)
        texts = texts.apply(self._preprocess)
        return Corpus(texts, metas=metas)

    def _build_lazy_metas(self, metas: dict):
        # build lazies
        lazies = (mc for mc in self._meta_configs.values() if mc.lazy)
        lazy: MetaConfig
        for lazy in lazies:
            if type(lazy) == DateTimeMetaConfig:
                lazy: DateTimeMetaConfig
                read_func = partial(pd.read_csv, usecols=lazy.columns,
                                    parse_dates=lazy.get_parsed_dates(), infer_datetime_format=True)
            else:
                read_func = partial(pd.read_csv, usecols=[lazy.column], dtype={lazy.column: lazy.dtype}, sep=self._sep)
            metas[lazy.column] = SeriesMeta(lazy.column, LazySeries(self._paths, self._nrows, read_func))

        return metas

    def _build_series_meta_and_text(self, metas: dict):
        series_and_dtypes = {mc.column: mc.dtype for mc in self._meta_configs.values()
                             if not mc.lazy and type(mc) != DateTimeMetaConfig}
        series_and_dtypes[self._col_text] = pd.StringDtype('pyarrow')

        all_cols = set(series_and_dtypes.keys())
        parse_dates: DateTimeMetaConfig = self._meta_configs.get(DateTimeMetaConfig.COL_DATETIME, False)
        if parse_dates:
            all_cols = all_cols.union(set(parse_dates.columns))
            parse_dates: dict = parse_dates.get_parsed_dates()
        current = 0
        dfs = list()
        for path in self._paths:
            if self._nrows is None:
                df = pd.read_csv(path, nrows=self._nrows, usecols=all_cols, sep=self._sep,
                                 parse_dates=parse_dates, dtype=series_and_dtypes)
            else:
                if current >= self._nrows:
                    break
                df = pd.read_csv(path, nrows=self._nrows - current, usecols=all_cols, sep=self._sep,
                                 parse_dates=parse_dates, dtype=series_and_dtypes)
                current += len(df)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)

        if self._col_text not in df.columns:
            raise KeyError(f"{self._col_text} column is missing. This column is compulsory. "
                           f"Did you call {self.set_text_column.__name__}?")

        # set up corpus dependencies here
        series_text = df.loc[:, self._col_text]
        del series_and_dtypes[self._col_text]
        for col in series_and_dtypes.keys():
            series = df[col]
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please use a different column name.")
            metas[col] = SeriesMeta(col, series)
        return metas, series_text


if __name__ == '__main__':
    from pathlib import Path

    builder = CorpusBuilder([Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'),
                             Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_1.csv')])
    builder.set_nrows(100)
    builder.set_sep(',')
    # cb.set_text_column('processed_text')
    # for col in ['year', 'day', 'tweet_lga', 'lga_code_2020']:
    #     cb.add_meta(col, lazy=True)
    # cb.add_meta('month', dtype='string', lazy=False)

    builder.set_text_column('processed_text')
    # builder.add_metas('created_at', dtypes='datetime', lazy=True)
    builder.add_metas(['geometry', 'state_name_2016'], dtypes=['object', 'str'])

    print(builder.show_columns())
    corpus = builder.build()
    print(corpus.metas())
    # print(corpus.get_meta('tweet_lga').preview(5))
    # print(corpus.get_meta('created_at').head(5))
