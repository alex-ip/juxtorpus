import pandas as pd
import pathlib
from pathlib import Path
from functools import partial
from typing import Union, Callable, Optional, Generator
from IPython.display import display
from ipywidgets import HBox, HTML, Layout, Text, Button, Output, VBox, Label, Checkbox, Dropdown

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.meta import SeriesMeta
from juxtorpus.viz import Widget
from juxtorpus.viz.widgets.corpus.builder import CorpusBuilderWidget
from juxtorpus.loader import LazySeries
from juxtorpus.utils.utils_pandas import row_concat

import logging

logger = logging.getLogger(__name__)

PROMPT_MISMATCHED_COLUMNS = "There are mismatched columns. These will be filled with NaN. " \
                            "Would you like to proceed? (y/n): "
PROMPT_MISMATCHED_COLUMNS_PASS = 'y'


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
        if isinstance(columns, str) or len(columns) == 1:
            column = columns if isinstance(columns, str) else columns[0]
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


class CorpusBuilder(Widget):
    """ CorpusBuilder

    The CorpusBuilder is used to construct a Corpus object. It turns tabular data from disk (currently only csv) to
    the corpus object with the aim of making it easier to construct a well-formed Corpus.

    This class follows the Builder pattern.
    Most exposed functions are to set up the correct state and then `build()` the corpus based on those states.

    ```Example Usage:
    builder = CorpusBuilder(paths)
    builder.add_metas(['your_meta_0', 'your_meta_1'], dtypes='category')
    builder.add_metas('year_month_day', dtypes='datetime')  # this will keep meta id as 'year_month_day'
    builder.set_text_column('text')
    corpus = builder.build()
    ```
    """

    allowed_dtypes = SeriesMeta.dtypes.union({'datetime'})

    def __init__(self, paths: Union[str, pathlib.Path, list[pathlib.Path]]):
        if type(paths) not in (list, set): paths = [paths]
        if type(paths) == Generator: paths = list(paths)
        for i in range(len(paths)):
            paths[i] = Path(paths[i])
        self._paths = paths
        self._nrows = None
        self._meta_configs = dict()
        self._sep = ','
        self._col_text = None
        self._dtype_text = pd.StringDtype('pyarrow')
        self._name = None

        # validate column alignments
        self._columns = self._prompt_validated_columns(self._paths)
        if self._columns is None: return

        self._preprocessors = list()

        self._widget = CorpusBuilderWidget(self)

    def _prompt_validated_columns(self, paths: list[pathlib.Path]) -> Optional[list[str]]:
        columns = list()
        for path in paths:
            name = path.stem
            if len(name) > 10: name = path.stem[:4] + '..' + path.stem[-4:]
            columns.append(pd.Series('✅', index=self.read(path, nrows=0).columns, name=name))
        df_cols = pd.concat(columns, axis=1)
        if df_cols.isnull().values.any():
            display(df_cols.fillna(''))
            if not input(PROMPT_MISMATCHED_COLUMNS).strip() == PROMPT_MISMATCHED_COLUMNS_PASS: return None
        return sorted(df_cols.index.to_list())

    @property
    def paths(self):
        return self._paths

    @property
    def columns(self):
        return self._columns

    def head(self, n: int = 3, cols: Optional[Union[str, list[str]]] = None):
        if cols is None: cols = self.columns
        return self.read(self._paths[0], nrows=n, usecols=cols)

    def summary(self):
        all_cols = pd.Series(sorted(list(self._columns)), name='All Columns')
        df = pd.DataFrame(index=all_cols, columns=['Text', 'Meta', 'Dtype'])
        df['Text'] = ''
        df['Meta'] = ''
        df['Dtype'] = ''

        # Populate Text column
        if self.text_column_is_set():
            df.loc[self._col_text, 'Text'] = '✅'
            df.loc[self._col_text, 'Dtype'] = str(self._dtype_text)

        # Populate Meta column
        for mc in self._meta_configs.values():
            if type(mc) == DateTimeMetaConfig and mc.is_multi_columned():
                for col in mc.columns:
                    df.loc[col, 'Meta'] = '✅'
            else:
                df.loc[mc.column, 'Meta'] = '✅'

        # Populate dtype column
        for row in df.itertuples(index=True):
            if not row.Meta: continue
            dtype = self._meta_configs.get(row.Index).dtype
            dtype = dtype if dtype is not None else 'inferred'
            df.loc[row.Index, 'Dtype'] = dtype

        df.sort_index(axis=0, inplace=True)
        return df.sort_index(axis=0, ascending=True).T

    def add_metas(self, columns: Union[str, list[str]],
                  dtypes: Union[None, str, list[str]] = None,
                  lazy=True):
        """ Add a column to add as metadata OR a list of columns to add.

        :param columns: The columns to add to the corpus.
        :param dtypes: The dtypes to specify.
        :param lazy: Keep series on disk until required.(Default: True)

        If dtype is None, dtype is inferred by pandas.
        """
        if isinstance(columns, str):
            columns = [columns]
        if dtypes == 'datetime':
            self._add_datetime_meta(columns, lazy)
            return
        if isinstance(dtypes, str) or dtypes is None:
            dtypes = [dtypes] * len(columns)
        if len(columns) != len(dtypes): raise ValueError("Number of columns and dtypes must match.")
        for i in range(len(columns)):
            col, dtype = columns[i], dtypes[i]
            if col == self._col_text: self._col_text = None  # reset text column
            if dtype is not None and dtype not in self.allowed_dtypes:
                raise ValueError(f"{dtype} is not a valid dtype.\nValid dtypes: {sorted(self.allowed_dtypes)}")
            else: self._add_meta(col, dtype, lazy)

    def _add_meta(self, column: str, dtype: Optional[str], lazy: bool):
        if column not in self._columns: raise ValueError(f"{column} column does not exist.")
        meta_config = MetaConfig(column=column, dtype=dtype, lazy=lazy)
        self._meta_configs[meta_config.column] = meta_config

    def _add_datetime_meta(self, columns: Union[str, list[str]], lazy: bool):
        if isinstance(columns, list) and len(columns) > 1:
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

    def set_document_column(self, column: str):
        if column not in self._columns:
            raise KeyError(
                f"Column: '{column}' not found. Use {self.summary.__name__} to preview the columns in the dataframe")
        self._col_text = column
        self._meta_configs.pop(column, None)

    def text_column_is_set(self):
        """ Text column is set. """
        return self._col_text is not None

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

    def set_name(self, name: str):
        self._name = name

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
        if not self.text_column_is_set():
            raise ValueError(f"You must set the text column. Try calling {self.set_document_column.__name__} first.")
        metas = dict()
        metas = self._build_lazy_metas(metas)
        metas, texts = self._build_series_meta_and_text(metas)
        texts = texts.apply(self._preprocess)
        return Corpus(texts, metas=metas, name=self._name)

    def _build_lazy_metas(self, metas: dict):
        # build lazies
        lazies = (mc for mc in self._meta_configs.values() if mc.lazy)
        lazy: MetaConfig
        for lazy in lazies:
            if isinstance(lazy, DateTimeMetaConfig):
                lazy: DateTimeMetaConfig
                read_func = partial(self.read, usecols=[lazy.column])
                is_datetime = True
            else:
                dtype = {lazy.column: lazy.dtype} if lazy.dtype is not None else None
                read_func = partial(self.read, usecols=[lazy.column], dtype=dtype)
                is_datetime = False
            metas[lazy.column] = SeriesMeta(lazy.column, LazySeries(self._paths, self._nrows, read_func, is_datetime))
        return metas

    def _build_series_meta_and_text(self, metas: dict):
        datetime_cols = [mc.column for mc in self._meta_configs.values()
                         if not mc.lazy and isinstance(mc, DateTimeMetaConfig)]
        col_to_dtype_map = {mc.column: mc.dtype for mc in self._meta_configs.values()
                            if not mc.lazy and not isinstance(mc, DateTimeMetaConfig)}
        col_to_dtype_map[self._col_text] = self._dtype_text
        all_cols = set(col_to_dtype_map.keys()).union(set(datetime_cols))

        current = 0
        dfs = list()
        for path in self._paths:
            if self._nrows is None:
                df = self.read(path, nrows=self._nrows, usecols=all_cols, dtype=col_to_dtype_map)
            else:
                if current >= self._nrows:
                    break
                df = self.read(path, nrows=self._nrows - current, usecols=all_cols, dtype=col_to_dtype_map)
                current += len(df)
            dfs.append(df)
        df = row_concat(dfs, ignore_index=True)

        for col in datetime_cols: df[col] = pd.to_datetime(df[col])
        if self._col_text not in df.columns:
            raise KeyError(f"{self._col_text} column is missing. This column is compulsory. "
                           f"Did you call {self.set_document_column.__name__}?")

        # set up corpus dependencies here
        series_text = df.loc[:, self._col_text]
        del col_to_dtype_map[self._col_text]
        all_cols = all_cols.difference({self._col_text})
        for col in all_cols:
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please use a different column name.")
            series = df[col]
            metas[col] = SeriesMeta(col, series)
        return metas, series_text

    def read(self, path: Path, nrows=None, usecols=None, dtype=None, **kwargs) -> pd.DataFrame:
        if path.suffix == '.csv':
            return pd.read_csv(path, nrows=nrows, usecols=usecols, dtype=dtype, **kwargs)
        elif path.suffix in ('.xlsx', '.xls'):
            return pd.read_excel(path, nrows=nrows, usecols=usecols, dtype=dtype, **kwargs)
        else:
            raise NotImplementedError("Only .csv and .xlsx are supported.")

    def set_callback(self, callback):
        self._widget.set_callback(callback)

    def widget(self):
        return self._widget.widget()
