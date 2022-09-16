import pathlib
import unittest


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_build_corpus_with_csv(self):
        # full procedural code to build corpus.
        from juxtorpus.corpus import Corpus, TweetCorpus

        # 1. from disk
        # 2. load csv for the columns
        # 3. set up meta objects + lazy series
        # 4. set up dataframe
        # -2. dataframe, metas
        # -1. corpus
        print()
        path: str = '~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv'
        nrows: int = 100
        corpus_cls = Corpus

        # META DATA #
        # load the dataframe, column first.
        import pandas as pd
        first_row = pd.read_csv(path, nrows=1)
        columns = first_row.columns.tolist()
        print(first_row.info(memory_usage='deep'))

        # dissect all the metadata and create the meta objects
        ## check for main column existance
        column_text: str = 'text'
        if column_text not in columns: raise KeyError(f"{column_text} not found.")
        columns.remove(column_text)
        ## put all metadata into lazyseries
        from juxtorpus.meta import LazyCSVSeries, Meta, CategoricalSeriesMeta, SeriesMeta
        from typing import List

        metas: List[Meta] = list()
        for meta_col in columns:
            ### recognise dtype
            dtype = None
            meta_cls = SeriesMeta
            if meta_col in ('tweets_lga'):
                dtype = 'categorical'
                meta_cls = CategoricalSeriesMeta
            series = LazyCSVSeries(path=path, col=meta_col, nrows=nrows, dtype=dtype)
            ## meta factory?
            metas.append(meta_cls(meta_col, series))

        # ACTUAL TEXT #
        corpus_df = LazyCSVSeries(path=path, col=column_text, nrows=nrows, dtype=None).load().to_frame(column_text)

        # construct the Corpus
        corpus = corpus_cls(corpus_df, metas)
        print(corpus.preprocess().summary())

    def test_build_corpus_with_text(self):
        from juxtorpus.corpus import Corpus, TweetCorpus

        # 1. from disk
        # 2. load csv for the columns
        # 3. set up meta objects + lazy series
        # 4. set up dataframe
        # -2. dataframe, metas
        # -1. corpus
        print()
        path: str = '~/Downloads/Geo_texts.txt'
        nrows: int = 100
        corpus_cls = Corpus

        # META DATA #

        # ACTUAL TEXT #

        import pandas as pd
        from juxtorpus.meta import LazyDataFrame
        column_text: str = 'text'
        read_func = lambda: pd.read_table(path, nrows=nrows, names=[column_text])
        df = LazyDataFrame(read_func).load()

        corpus_df = corpus_cls(df)
        print(corpus_df.preprocess().summary())

    def test_build_corpus_with_texts(self):
        """ Multiple text and meta files."""

        # 1. from disk
        # 2. load csv for the columns
        # 3. set up meta objects + lazy series
        # 4. set up dataframe
        # -2. dataframe, metas
        # -1. corpus
        print()
        from juxtorpus.corpus import Corpus
        from pathlib import Path
        from typing import Generator
        import os
        paths: Generator[pathlib.PosixPath, None, None] = Path(f"{os.getenv('HOME')}/Downloads").glob('Geo_texts*.txt')
        nrows: int = 1000
        corpus_cls = Corpus

        # META DATA #

        # ACTUAL TEXT #

        import pandas as pd
        from juxtorpus.meta import LazyDataFrame
        column_text: str = 'text'

        rows_loaded = 0
        rows_to_load = nrows
        dfs = list()
        for p in paths:
            print(f"++ Reading from {p}...")
            read_func = lambda: pd.read_table(p, nrows=rows_to_load, names=[column_text])
            df = LazyDataFrame(read_func).load()
            dfs.append(df)
            rows_loaded += len(df)
            if rows_loaded >= nrows:
                break
            else:
                rows_to_load = nrows - rows_loaded

        # todo: keep as lazy dfs?
        # lazy_dfs = list()
        # for p in paths:
        #     read_func = lambda: pd.read_table(p)
        #     lazy_dfs.append(LazyDataFrame(read_func))

        df = pd.concat(dfs, axis=0, names=[column_text], copy=False)

        assert len(df) == nrows, "The expected number of rows to load is incorrect."

        c = corpus_cls(df=df)
        print(c.preprocess().summary())
