""" Item Timeline

Plot a number of items and their chosen metric across a shared timeline.
"""
import pandas as pd

"""
NOTES:
Trace:
Inputs: list of timestamps, list of counts/metric
Output: a Trace


Timeline:
Factory Methods:
1. from_freqtables
2. from_corpus_groups ---> defaults to using the standard dtm

Features:
1. Mode: Highlight Peaks, Highlight Cumulative
2.


Properties:
1. expose terms
2. expose terms and their colours
"""
import pandas as pd
from pandas.api.types import is_datetime64_dtype
import numpy as np
from typing import Union, Optional
import plotly.graph_objs as go
from collections import namedtuple
from functools import partial
import colorsys
import random

from juxtorpus.viz import Viz
from juxtorpus.corpus.freqtable import FreqTable

TNUMERIC = Union[int, float]
TPLOTLY_RGB_COLOUR = str


class ItemTimeline(Viz):
    TIMELINE_TRACE = namedtuple('TIMELINE_TRACE', ['item', 'datetimes', 'metrics', 'colour'])

    @classmethod
    def from_freqtables(cls, datetimes: Union[pd.Series, list], freqtables: list[FreqTable]):
        """ Constructs ItemTimeline using the specified freqtables"""
        if len(datetimes) != len(freqtables):
            raise ValueError(f"Mismatched length of datetimes and freqtables. {len(datetimes)} and {len(freqtables)}.")
        fts_df = pd.concat([ft.series for ft in freqtables], axis=1).fillna(0).T
        datetimes = pd.to_datetime(pd.Series(datetimes))
        fts_df.reset_index(drop=True)
        fts_df.set_index(datetimes, inplace=True)
        return cls(df=fts_df)

    @classmethod
    def from_corpus_groups(cls, groups):
        """ Constructss ItemTimeline from corpus groups. Note: Default DTM is used as the items. """
        groups = list(groups)
        fts = [c.dtm.freq_table() for _, c in groups]
        datetimes = [dt for dt, _ in groups]
        return cls.from_freqtables(datetimes, fts)

    def __init__(self, df: pd.DataFrame):
        """ Initialise with a dataframe with a datetime index, item columns and values as metrics. """
        self._df: pd.DataFrame = df
        assert is_datetime64_dtype(self._df.index), "DataFrame Index must be datetime."
        self.datetimes = self._df.index.to_list()

        self.MODE_PEAK = 'PEAK'
        self.MODE_CUMULATIVE = 'CUMULATIVE'
        self.modes = {
            self.MODE_PEAK: partial(pd.DataFrame.max, axis=0),  # across datetime
            self.MODE_CUMULATIVE: partial(pd.DataFrame.sum, axis=0)
        }
        self.DEFAULT_MODE = self.MODE_PEAK
        self.mode = None
        self._metric_series = None
        self.items = None
        self.set_mode(self.DEFAULT_MODE)

        self.DEFAULT_TOP = 3
        self.top = None
        self.set_top(3)

        self.seed(42)
        self._rint = random.randint

    @staticmethod
    def seed(seed: int):
        """ Set the seed across all item timeline objects. """
        random.seed(seed)

    def set_mode(self, mode: Optional[str]):
        if mode is None:
            self.mode = None
            self.items = self._df.columns.to_list()
        else:
            mode = mode.upper()
            if mode not in self.modes.keys(): raise ValueError(f"{mode} not in {', '.join(self.modes.keys())}")
            self.mode = mode
            self._metric_series: pd.Series = self.modes.get(self.mode)(self._df)
            # critical functions for top
            self._metric_series.sort_values(ascending=False, inplace=True)
            self.items = self._metric_series.index.to_list()

    def set_top(self, top: int):
        self.top = top

    def render(self):
        fig = self._build()
        fig.show()

    def _build(self):
        fig = go.Figure()
        for i, item in enumerate(self.items):
            trace = ItemTimeline.TIMELINE_TRACE(item=item, datetimes=self.datetimes, metrics=self._df.loc[:, item],
                                                colour=self._get_colour(item))
            fig.add_trace(
                go.Scatter(
                    x=trace.datetimes, y=trace.metrics,
                    mode='lines+markers', marker_color=trace.colour,
                    name=trace.item
                )
            )
        return fig

    def _get_colour(self, item):
        r, g, b = self._get_rgb(item)
        opacity = self._get_opacity(item)
        return f'rgba({r},{g},{b},{opacity})'

    def _get_rgb(self, item: str) -> TPLOTLY_RGB_COLOUR:
        rint = self._rint
        h = hash(item)
        return (h * rint(0, 10)) % 256, (h * rint(0, 10)) % 256, (h * rint(0, 10)) % 256

    def _get_opacity(self, item):
        # no modes selected
        if self.mode is None: return 1.0
        else:
            # top
            idx = self._metric_series.index.get_loc(item)
            if idx < self.top: return 1.0

            # gradient
            metric = self._metric_series.loc[item]
            if metric > self._metric_series.quantile(0.9): return 0.2
            return 0.01
