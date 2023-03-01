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
import ipywidgets as widgets
from IPython.display import display

from juxtorpus.viz import Viz
from juxtorpus.corpus.freqtable import FreqTable

TNUMERIC = Union[int, float]
TPLOTLY_RGB_COLOUR = str


class ItemTimeline(Viz):
    TRACE_DATUM = namedtuple('TRACE_DATUM', ['item', 'datetimes', 'metrics', 'colour'])

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
        self.mode = self.DEFAULT_MODE
        self._metric_series = None
        self.items = None

        # top items
        self.DEFAULT_TOP = 30
        self.top = self.DEFAULT_TOP

        self._update_metrics(self.mode, self.top)

        # opacity
        self.FULL_OPACITY_TOP = 3  # top number of items with full opacity

        self.seed(42)
        self._rint = random.randint

    @staticmethod
    def seed(seed: int):
        """ Set the seed across all item timeline objects. """
        random.seed(seed)

    def set_mode(self, mode: Optional[str]):
        """ Sets the mode of the timeline as 'Peak' or 'Cumulative'. """
        # updates the items to display.
        if mode is None:
            self.mode = None
            self.items = self._df.columns.to_list()
        else:
            mode = mode.upper()
            if mode not in self.modes.keys(): raise ValueError(f"{mode} not in {', '.join(self.modes.keys())}")
            self.mode = mode
            self._update_metrics(self.mode, self.top)
            # critical functions for top
            # self._metric_series.sort_values(ascending=False, inplace=True)
            # self.items = self._metric_series.index.to_list()

    def set_top(self, top: int):
        if top < 1: raise ValueError(f"Must be > 1.")
        self.top = top
        self._update_metrics(self.mode, self.top)

    def _update_metrics(self, mode: str, top: int):
        metric_series = self.modes.get(mode)(self._df)
        metric_series.sort_values(ascending=False, inplace=True)
        metric_series = metric_series.iloc[:top]
        self._metric_series = metric_series
        self.items = self._metric_series.index.to_list()

    def render(self):
        fig = self._build()
        button = self._create_dropdown_widget(fig)
        display(button)
        return fig

    def _build(self):
        fig = go.FigureWidget()
        for tdatum in self._generate_trace_data():
            fig.add_trace(
                go.Scatter(
                    x=tdatum.datetimes, y=tdatum.metrics,
                    mode='lines+markers+text', marker_color=tdatum.colour,
                    name=tdatum.item,
                )
            )

        self._add_toggle_all_selection_layer(fig)
        self._add_top_items_slider_layer(fig)
        return fig

    def _generate_trace_data(self):
        """ Generates the trace content data from the current state (mode, top)"""
        trace_data = []
        for i, item in enumerate(self.items):
            tdatum = ItemTimeline.TRACE_DATUM(item=item, datetimes=self.datetimes, metrics=self._df.loc[:, item],
                                              colour=self._get_colour(item))
            trace_data.append(tdatum)
        return trace_data

    def _update_traces(self, fig):
        trace_data = self._generate_trace_data()
        with fig.batch_update():
            for i, trace in enumerate(fig.data):
                tdatum = trace_data[i]
                trace.name = f'{tdatum.item}'
                trace.y = tdatum.metrics

    def _create_dropdown_widget(self, fig):
        dropdown = widgets.Dropdown(
            options=[mode.capitalize() for mode in sorted(list(self.modes.keys()))],
            value=self.mode.capitalize(),
            description='Mode: ',
            disabled=False
        )

        def observe_dropdown(event):
            self.set_mode(dropdown.value.upper())
            self._update_traces(fig)

        dropdown.observe(observe_dropdown)
        return dropdown

    @staticmethod
    def _add_toggle_all_selection_layer(fig):
        """ Adds a layer to select/deselect all the traces of the timeline. """
        fig.update_layout(dict(updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=["visible", True],
                        label="Select All",
                        method="restyle",
                    ),
                    dict(
                        args=[{"visible": ['legendonly'] * len(fig.data)}],
                        label="Deselect All",
                        method="restyle",
                    ),
                ]),
                pad={"r": 1, "t": 1},
                showactive=False,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top",
                font=dict(size=12)
            ),
        ]
        ))

    def _add_top_items_slider_layer(self, fig):
        steps = []
        for i in reversed(range(len(fig.data))):
            step = dict(
                method='update',
                args=[{'visible': [True if j <= i else 'legendonly' for j in range(len(fig.data))]},
                      {'title': self._get_title(i + 1)}],
                label=f'{i + 1}',
            )
            steps.append(step)

        sliders = [dict(
            active=self.top,
            currentvalue={'prefix': 'Top: '},
            pad={'t': 25},
            steps=steps
        )]
        # pad = {'t': 20}
        fig.update_layout(sliders=sliders)
        return fig

    @staticmethod
    def _add_toggle_all_selection_layer(fig):
        """ Adds a layer to select/deselect all the traces of the timeline. """
        fig.update_layout(dict(updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": ['legendonly'] * len(fig.data)}],
                        label="Deselect All",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", True],
                        label="Select All",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
        ))
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
            if idx < self.FULL_OPACITY_TOP: return 1.0

            # gradient
            metric = self._metric_series.loc[item]
            if metric > self._metric_series.quantile(0.5): return 0.4
            return 0.1

    def _get_title(self, i):
        return f'Top {i} {self.mode.capitalize()} items'
