from abc import ABC
from typing import Union, Callable

import pandas as pd
from ipywidgets import SelectMultiple, VBox, HBox, Select, Layout, Label, Button, Checkbox, HTML, Text
from ipywidgets import Widget as ipyWidget
from copy import deepcopy

from juxtorpus.viz import Widget
from juxtorpus.interfaces import Container
from juxtorpus.corpus.meta import SeriesMeta
from juxtorpus.viz.style.ipyw import *

TMASK = 'pd.Series[bool]'

TYPES = ['category', 'datetime', 'whole number', 'decimal', 'text']


class _Operation(object):
    """ Operation
    There can be category, datetime, numeric, text operations.
    What we need from each:
    category -> item name - filter_by_item
    datetime -> start, end - filter_by_datetime
    whole number -> integer - filter_by_item, filter_by_range
    decimal -> float - filter_by_item, filter_by_range
    text -> str - filter_by_item, filter_by_regex
    """

    def __init__(self, meta_id: str, type_: str, **kwargs):
        assert type_ in set(TYPES), f"{type_} must be one of {', '.join(TYPES)}"
        self._meta_id = meta_id
        self._type = type_
        self._config = kwargs

    @property
    def meta_id(self):
        return self._meta_id

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config


class _Operations(dict, Widget, ABC):
    """ Store the Slicing Operations and shows a Checkbox List Widget to allow selection."""

    def __init__(self):
        super().__init__(self)
        self._label = Label("Operations", layout=Layout(**center_text))
        self._button = Button(description="Add Operation", layout=Layout(**center_text))
        self._vbox = VBox()

        self._checkbox_toggled_callbacks = list()

    def button(self):
        return self._button

    def preview(self, corpus: 'Corpus') -> int:
        """ Return the size of the Corpus only after slicing. """
        pass

    def apply(self, corpus: 'Corpus') -> 'Corpus':
        """ Apply selected operations on Corpus. """
        pass

    def widget(self):
        """ This widget should show all the operations that are added to this container. """
        vbox = VBox()
        for cb, ops in self._ops.items():
            pass

    def _create_checkbox(self, meta_id: str, op: _Operation) -> Checkbox:
        desc = f"[{meta_id}] {op.config}"
        cb = Checkbox(description=desc, layout=Layout(**no_horizontal_scroll))
        cb.style = {'description_width': '0px'}

        # when checkbox is toggled, this should create the mask for the subcorpus.
        def observe(event):
            # todo: slice to mask of all toggled operation, then run the callbacks.
            for callback in self._checkbox_toggled_callbacks: callback()

        cb.observe(observe, names='value')
        return cb

    def add_checkbox_toggled_callback(self, funcs: Union[Callable, list[Callable]]):
        funcs = list(funcs)
        self._checkbox_toggled_callbacks.extend(funcs)

    def _observe_button(self, event):
        # create checkbox and add to vbox.
        # self._create_checkbox()
        # this requires the CURRENT operation and add that as dropdown?
        # for operations not added into the VBox, Add that in?
        self._vbox.children = (self._vbox.children,)  # *current op


class SlicerWidget(Widget, ABC):

    def __init__(self, corpus: 'Corpus'):
        self.corpus = corpus
        # self._dashboard = None
        self._sliced_mask: TMASK = None
        self._ops = _Operations()
        self._ops_tmp = _Operations()

        self._header_selector = Label('Meta', layout=Layout(**no_horizontal_scroll, **center_text))
        self._header_filter = Label('Filter By', layout=Layout(**no_horizontal_scroll, **center_text))
        self._header_preview = Label("Corpus Size", layout=Layout(**no_horizontal_scroll, **center_text))

    def widget(self):
        return self._dashboard()

    def _dashboard(self):
        """ Creates the full dashboard. """
        panels = self._panels()
        selector_box = self._selector_box(panels)
        panel_box = self._panel_box(panels)
        slice_box = self._slice_box()

        return HBox([selector_box, panel_box, slice_box],
                    layout=Layout(**no_horizontal_scroll))

    def _selector_box(self, panels: dict[str, ipyWidget]) -> Select:
        meta_ids = [opt for opt in panels.keys()]
        return Select(
            options=meta_ids,
            value=meta_ids[0] if len(meta_ids) > 0 else None,
            disabled=False, layout=Layout(**no_horizontal_scroll)
        )

    def _panel_box(self, panels: dict[str, ipyWidget]) -> VBox:
        """ Returns the panel box. """
        return VBox([panel for panel in panels.values()],
                    layout=Layout(**no_horizontal_scroll))

    def _slice_box(self, ):
        """ Return a VBox of Corpus Preview + Corpus ID text box"""
        preview = HTML(f"<h2>xx</h2>")
        corpus_id = Text(placeholder='Corpus ID')
        slice_button = Button(description='Slice')
        return VBox([preview, corpus_id, slice_button], layout=Layout(**no_horizontal_scroll))

    def _panels(self) -> dict[str, ipyWidget]:
        """ Create all the slicing panels. """
        panels = dict()
        for meta_id, meta in self.corpus.meta.items():
            if not isinstance(meta, SeriesMeta):
                raise NotImplementedError(f"SliceWidget currently only supports {SeriesMeta.__class__.__name__}.")

            dtype = meta.dtype
            if dtype == 'category':
                panel = self._category_panel(meta)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                panel = self._datetime_panel()
            elif pd.api.types.is_integer_dtype(dtype):
                panel = self._wholenumber_panel()
            elif pd.api.types.is_float_dtype(dtype):
                panel = self._decimal_panel()
            elif pd.api.types.is_string_dtype(dtype):
                panel = self._text_panel()
            else:
                raise NotImplementedError(f"No slicer panels for {dtype=}")
            panels[meta_id] = panel
        return panels

    def _category_panel(self, meta: SeriesMeta) -> SelectMultiple:
        panel = SelectMultiple(
            options=sorted((str(item) for item in meta.series.unique().tolist())),
            layout=Layout(**no_horizontal_scroll)
        )

        def observe(_):
            # when the option is selected, update the operation associated with meta
            op = _Operation(type_='category', item=panel.value)
            self._ops[meta.id] = op

        panel.observe(observe, names='value')
        return panel

    def _datetime_panel(self) -> VBox:
        pass

    def _wholenumber_panel(self) -> VBox:
        pass

    def _decimal_panel(self) -> VBox:
        pass

    def _text_panel(self) -> VBox:
        pass
