from abc import ABC
from typing import Union, Callable

import pandas as pd
from ipywidgets import (
    SelectMultiple, VBox, HBox, Select, Layout, Label, Button, Checkbox,
    HTML, Text, DatePicker, SelectionRangeSlider, ToggleButtons, Box, IntText, FloatText
)

from ipywidgets import Widget as ipyWidget
from copy import deepcopy
from datetime import timedelta, datetime

from juxtorpus.corpus.operations import Operations, Operation
from juxtorpus.viz import Widget
from juxtorpus.corpus.meta import SeriesMeta
from juxtorpus.viz.style.ipyw import *
from juxtorpus.viz.widgets.corpus.operations import OperationsWidget

TMASK = 'pd.Series[bool]'

TYPES = ['category', 'datetime', 'whole number', 'decimal', 'text']

STRFORMAT = ' %d %b %Y '


class SlicerWidget(Widget, ABC):
    class _ConfigState(dict):
        def __init__(self, metas):
            super().__init__()
            for meta_id in metas.keys(): self[meta_id] = dict()
            self.selected_meta = list(metas.keys())[0]

    def __init__(self, corpus: 'Corpus'):
        self.corpus = corpus
        # self._dashboard = None
        self._sliced_mask: TMASK = None
        self._ops = Operations()  # start with empty operations.
        self._ops_widget = OperationsWidget(self.corpus, self._ops)

        self._header_selector = Label('Meta', layout=Layout(**no_horizontal_scroll, **center_text))
        self._header_filter = Label('Filter By', layout=Layout(**no_horizontal_scroll, **center_text))
        self._header_preview = Label("Corpus Size", layout=Layout(**no_horizontal_scroll, **center_text))

        # internal mutable states
        self._state: SlicerWidget._ConfigState = self._ConfigState(corpus.meta)
        self._panels_: dict[str, ipyWidget] = self._panels()
        self._panel_box_ = self._panel_box(selected_meta=list(self._panels_.keys())[0])
        self._dashboard_ = self._dashboard()

    def widget(self):
        return self._dashboard_

    def _dashboard(self):
        """ Creates the full dashboard. """
        selector_box = self._selector_box(self._panels_)
        add_operation_btn = self._add_operation_button()
        top = HBox([selector_box, self._panel_box_, add_operation_btn], layout=Layout(**no_horizontal_scroll))
        bottom = Label()  # placeholder, before Add Operation button is clicked.
        return VBox([top, bottom], layout=Layout(**no_horizontal_scroll))

    def _selector_box(self, panels: dict[str, ipyWidget]) -> Select:
        meta_ids = [meta_id for meta_id in panels.keys()]
        select = Select(
            options=meta_ids,
            value=meta_ids[0] if len(meta_ids) > 0 else None,
            disabled=False, layout=Layout(height='100%', width='33%')
        )

        def observe(_):
            self._state.selected_meta = select.value
            self._update_panel_box_with(select.value)

        select.observe(observe, names='value')
        return select

    def _panel_box(self, selected_meta) -> VBox:
        """ Returns the panel box. """
        return VBox([self._panels_.get(selected_meta)],
                    layout=Layout(**no_horizontal_scroll))

    def _update_panel_box_with(self, selected_meta):
        self._panel_box_.children = (self._panels_.get(selected_meta),)

    def _add_operation_button(self) -> Button:
        button = Button(description='Add Operation')

        # onclick - Add to ops, refresh bottom panel
        def on_click(event):
            self._ops.add(self._create_operation_for_selected())
            self._refresh_ops_widget()

        button.on_click(on_click)
        return button

    def _create_operation_for_selected(self) -> Operation:
        selected_meta = self._state.selected_meta
        meta = self.corpus.meta.get(selected_meta)
        config = self._state.get(selected_meta)
        # todo: create operation from meta and config.
        return Operation()

    def _refresh_ops_widget(self):
        """ Replaces the bottom of the dashboard with new ops widget. """
        self._dashboard_.children = (self._dashboard_.children[0], self._ops_widget.widget())

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
                panel = self._datetime_panel(meta)
            elif pd.api.types.is_integer_dtype(dtype):
                panel = self._wholenumber_panel(meta)
            elif pd.api.types.is_float_dtype(dtype):
                panel = self._decimal_panel(meta)
            elif pd.api.types.is_string_dtype(dtype):
                panel = self._text_panel(meta)
            else:
                raise NotImplementedError(f"No slicer panels for {dtype=}")
            panels[meta_id] = panel
        return panels

    def _category_panel(self, meta: SeriesMeta) -> SelectMultiple:
        panel = SelectMultiple(
            options=sorted((str(item) for item in meta.series.unique().tolist())),
            layout=Layout(**no_horizontal_scroll)
        )

        config = self._state.get(meta.id)

        def observe(_):
            # when the option is selected, update the configuration associated with meta
            config['item'] = panel.value

        panel.observe(observe, names='value')
        return panel

    def _datetime_panel(self, meta: SeriesMeta) -> VBox:
        series = meta.series
        margin = timedelta(days=1)
        start, end = series.min().to_pydatetime() - margin, series.max().to_pydatetime() + margin
        widget_s = DatePicker(description='start', value=start, layout=Layout(**{'width': '98%'}))
        widget_e = DatePicker(description='end', value=end, layout=Layout(**{'width': '98%'}))

        def date_to_slider_option(date) -> str:
            return date.strftime(STRFORMAT)

        def slider_option_to_date(option: str) -> datetime:
            return datetime.strptime(option, STRFORMAT)

        dates = pd.date_range(start, end, freq='D')
        options = [date_to_slider_option(date) for date in dates]
        index = (int(len(dates) / 4), int(3 * len(dates) / 4))
        slider = SelectionRangeSlider(options=options, index=index, layout={'width': '98%'})

        config = self._state.get(meta.id)

        def update_datetime_datepicker(event):
            sv, ev = widget_s.value, widget_e.value
            config['start'] = date_to_slider_option(sv) if sv is not None else options[0]
            config['end'] = date_to_slider_option(ev) if ev is not None else options[-1]
            try:
                # slider.index = (options.index(date_to_slider_option(widget_s.value)),
                #                 options.index(date_to_slider_option(widget_e.value)))
                slider.index = (options.index(config['start']),
                                options.index(config['end']))
            except ValueError as ve:
                # datepicker selected a date that's out of range from the slider.
                pass
            except Exception as e:
                # any other exceptions that can occur e.g. datepicker suddenly jumps to value = None
                pass

        def update_datetime_slider(event):
            config['start'] = slider.value[0]
            config['end'] = slider.value[1]
            widget_s.value = slider_option_to_date(config['start'])
            widget_e.value = slider_option_to_date(config['end'])

        update_datetime_slider(None)
        widget_s.observe(update_datetime_datepicker, names='value')
        widget_e.observe(update_datetime_datepicker, names='value')
        slider.observe(update_datetime_slider, names='value')
        return VBox([widget_s, widget_e, slider], layout=Layout(width='98%'))

    def _wholenumber_panel(self, meta) -> VBox:
        """ DType: whole number; filter_by_item, filter_by_range """
        return self._numeric_panel(meta, int)

    def _decimal_panel(self, meta) -> VBox:
        return self._numeric_panel(meta, float)

    def _numeric_panel(self, meta, type_) -> VBox:
        assert type_ in (int, float), "Type must be either int or float."
        WIDGET_NUM = IntText if type_ == int else FloatText
        config = self._state.get(meta.id)

        ft_min = WIDGET_NUM(description='Min:', layout=Layout(width='98%'), value=meta.series.min())
        ft_max = WIDGET_NUM(description='Max:', layout=Layout(width='98%'), value=meta.series.max())
        vbox_range = VBox([ft_min, ft_max], layout=Layout(width='98%'))
        ft_num = WIDGET_NUM(description='Number:', layout=Layout(width='98%'))
        box_num = Box([ft_num], layout=Layout(width='98%'))

        vboxes = [
            ('Min/Max', vbox_range),
            ('Number', box_num),
        ]
        starter_idx = 0
        options = [vbox[starter_idx] for vbox in vboxes]
        toggle = ToggleButtons(
            options=options, value=options[starter_idx], layout=Layout(**no_horizontal_scroll)
        )

        vbox = VBox([vboxes[starter_idx][1], toggle], layout=Layout(height='100%', **no_horizontal_scroll))

        # CALLBACKs
        config['mode'] = toggle.value
        config['number'] = ft_num.value
        config['range'] = dict()
        config['range']['min'] = ft_min.value
        config['range']['max'] = ft_max.value

        def observe_toggle(event):
            config['mode'] = event.get('new')
            alt_vbox = None
            for vb in vboxes:
                if vb[0] == config.get('mode'):
                    alt_vbox = vb[1]
                    break
            if alt_vbox is None: raise RuntimeError("This should not happen. Internal Error.")
            vbox.children = (alt_vbox, toggle)

        def observe_num(event):
            config['number'] = event.get('new')

        def observe_minmax(event):
            r = config.get('range')
            r['min'] = ft_min.value
            r['max'] = ft_max.value
            config['range'] = r

        toggle.observe(observe_toggle, names='value')
        ft_num.observe(observe_num, names='value')
        ft_min.observe(observe_minmax, names='value')
        ft_max.observe(observe_minmax, names='value')
        return vbox

    def _text_panel(self, meta) -> VBox:
        """ Dtype: text; filter_by_item, filter_by_regex """
        placeholder = meta.series.iloc[0]
        w_text = Text(placeholder=placeholder)
        w_toggle = ToggleButtons(
            options=['Text', 'Regex'],
            value='Text',
            description='',
            disabled=False,
            button_style='',
            tooltips=['Exact match', 'Enter a regex pattern'],
        )
        config = self._state.get(meta.id)
        config['mode'] = w_toggle.value

        def observe_toggle(event):
            config['mode'] = event.get('new')

        def observe_text(event):
            config['text'] = event.get('new')

        w_toggle.observe(observe_toggle, names='value')
        w_text.observe(observe_text, names='value')
        return VBox([w_text, w_toggle], layout=Layout(width='98%'))
