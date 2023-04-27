""" App

Registry holds all the corpus and sliced subcorpus in memory. Allowing on the fly access.
"""
import pandas as pd
# import numpy as np
from ipywidgets import Layout, Label, HBox, VBox, GridBox, Checkbox, SelectMultiple, \
    Box, Button, Select, DatePicker, Output, Text, HTML
import ipywidgets as widgets
from pathlib import Path
import math
from datetime import timedelta, datetime
import spacy
from copy import deepcopy

from juxtorpus.corpus.processors import process
from juxtorpus.corpus import Corpus, CorpusBuilder
from juxtorpus.corpus.meta import Meta, SeriesMeta
from juxtorpus.viz.widgets import FileUploadWidget

#
# python3.9/site-packages/traitlets/traitlets.py:697:
# FutureWarning: Comparison of Timestamp with datetime.date is deprecated in order to match the standard
# library behavior. In a future version these will be considered non-comparable.
# Use 'ts == pd.Timestamp(date)' or 'ts.date() == date' instead.
#   silent = bool(old_value == new_value)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def format_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# used in datepicker, and when slicing will use this for filter_by_datetime
STRFORMAT = ' %d %b %Y '


class App(object):
    DTYPES_MAP = {
        'auto': None,
        'decimal': 'float',
        'whole number': 'Int64',
        'text': 'str',
        'datetime': 'datetime',
        'category': 'category'
    }

    def __init__(self):
        self._CORPUS_SELECTOR_MAX_NUM_CORPUS_PER_COLUMN = 20  # TODO: link with _update_corpus_selector()
        self._corpus_selector_labels = [
            Label("Corpus ID", layout=_create_layout(**corpus_id_layout, **center_style)),
            Label("Size", layout=_create_layout(**size_layout, **center_style)),
            Label("Parent", layout=_create_layout(**parent_layout, **center_style))
        ]
        # registry
        self.REGISTRY = dict()
        self._selected_corpus: Corpus = None

        # corpus builder
        self._files: dict = dict()
        self._files_stats_df = None
        self._builder: CorpusBuilder = None

        # widgets
        self._corpus_builder_configs = None  # corpus_builder - stores all builder meta, text configs
        self._corpus_selector = None  # corpus_selector - registry
        self._corpus_slicer = None
        self._corpus_slicer_dashboard = None  # corpus_slicer - for referencing
        self._corpus_slicer_operations: dict = None  # corpus_slicer - stores all slicer operations.
        self._corpus_slicer_current_mask: pd.Series[bool] = None  # corpus_slicer - mask from all ops hist

    @property
    def corpus(self):
        return self._selected_corpus

    ## Corpus Registry ##
    def update_registry(self, corpus_id, corpus):
        if corpus_id in self.REGISTRY.keys(): raise KeyError(f"{corpus_id} already exists.")
        self.REGISTRY[corpus_id] = corpus
        self._update_corpus_selector()  # refreshes registry

    ## Widget: Corpus Builder ##
    def corpus_builder(self):
        # file uploader + files select
        f_selector = SelectMultiple(layout=_create_layout(**f_selector_layout))
        fuw = FileUploadWidget()
        fuw._uploader.layout = _create_layout(**f_uploader_layout)

        def _callback_fileupload_to_fileselector(fuw, added):
            self._files = {p.name: {'path': p} for p in fuw.uploaded()}
            f_selector.options = [name for name in self._files.keys()]
            print(self._files)

        fuw.set_callback(_callback_fileupload_to_fileselector)

        box_df = Box(layout=_create_layout(**box_df_layout))

        # hbox_corpus_builder = self._create_corpus_builder()

        def _observe_file_selected(event):
            # from pprint import pprint
            selected = event.get('new')
            for name, d in self._files.items():
                d['selected'] = True if name in selected else False

            # build files preview
            df_list = []
            for name, d in self._files.items():
                if d.get('selected'):
                    size = format_size(d.get('path').stat().st_size)
                    df_list.append((name, size))
            df = pd.DataFrame(df_list, columns=['name', 'size'])
            box_df.children = (widgets.HTML(df.to_html(index=False, classes='table')),)

        f_selector.observe(_observe_file_selected, names='value')
        button_confirm = Button(description='Confirm',
                                layout=Layout(width='20%', height='50px'))
        hbox_uploader = HBox([VBox([f_selector, fuw._uploader], layout=Layout(width='50%', height='200px')),
                              VBox([box_df, button_confirm], layout=Layout(width='50%', height='200px'))],
                             layout=Layout(width='100%', height='100%'))

        vbox = VBox([hbox_uploader, Box()])

        def on_click_confirm(_):
            selected_files = [d.get('path') for d in self._files.values() if d.get('selected')]
            if len(selected_files) <= 0: return
            # self._builder = CorpusBuilder(selected_files)
            # hbox_corpus_builder.children = tuple((Label(p.name) for p in self._builder.paths))
            hbox_corpus_builder = self._create_corpus_builder(selected_files)
            vbox.children = (vbox.children[0], hbox_corpus_builder)

        button_confirm.on_click(on_click_confirm)
        return vbox

    # Widget: Corpus Builder

    def _create_text_meta_dtype_row(self, id_: str, text_checked: bool, meta_checked: bool, config: dict):
        label = widgets.Label(f"{id_}", layout=widgets.Layout(width='30%'))
        t_checkbox = widgets.Checkbox(value=text_checked, layout=widgets.Layout(width='15%'))
        t_checkbox.style.description_width = '0px'

        m_checkbox = widgets.Checkbox(value=meta_checked, layout=widgets.Layout(width='15%'))
        m_checkbox.style.description_width = '0px'

        dtypes = sorted([k for k in self.DTYPES_MAP.keys()])
        dtype_dd = widgets.Dropdown(options=dtypes, value=dtypes[0], disabled=False,
                                    layout=widgets.Layout(width='100px'))

        # dtype_dd.observe # todo
        def _toggle_checkbox(event):
            if not event.get('new'):
                config['text'] = False
                config['meta'] = False
            else:
                if id(event.get('owner')) == id(t_checkbox):
                    m_checkbox.value = False
                    dtype_dd.value = dtypes[dtypes.index('text')]  # if ticked as text - sets dtype as str
                    config['text'] = True
                    config['meta'] = False
                elif id(event.get('owner')) == id(m_checkbox):
                    t_checkbox.value = False
                    dtype_dd.value = dtypes[dtypes.index('auto')]  # if ticked as text - sets dtype as str
                    config['meta'] = True
                    config['text'] = False

        def _update_dtype(event):
            config['dtype'] = self.DTYPES_MAP.get(event.get('new'))

        t_checkbox.observe(_toggle_checkbox, names='value')
        m_checkbox.observe(_toggle_checkbox, names='value')  # todo: toggle text and meta
        dtype_dd.observe(_update_dtype, names='value')
        return widgets.HBox([label, t_checkbox, m_checkbox, dtype_dd])

    def _create_corpus_builder(self, paths):
        # if self._builder is None: return VBox()  # self._builder must first be set up before.
        self._builder = CorpusBuilder(paths)

        # creates the top labels.
        top_labels = [('id', '30%'), ('document', '15%'), ('meta', '15%'), ('data type', '30%')]
        selection_top_labels = HBox(
            list(map(lambda ls: widgets.HTML(f"<b>{ls[0]}</b>", layout=Layout(width=ls[1])), top_labels))
        )

        # creates all the rows
        columns = self._builder.columns
        self._corpus_builder_configs = {col: {'text': False, 'meta': True} for col in columns}
        configs = self._corpus_builder_configs
        selection_widgets = [selection_top_labels]
        gen_rows = (self._create_text_meta_dtype_row(col, check.get('text'), check.get('meta'), check)
                    for col, check in configs.items())
        selection_widgets.extend(gen_rows)

        # create build button
        key_textbox = Text(description='Corpus ID:', placeholder='ID to be stored in the registry')
        corpus_id = {'name': ''}

        button = Button(description='Build')
        button_output = Output(layout=Layout(overflow='scroll hidden'))

        def _on_click_key_textbox(event):
            corpus_id.update({'name': event.get('new')}, names='value')
            button.disabled = len(corpus_id.get('name')) <= 0

        def _on_click_build_corpus(_):
            for key, config in configs.items():
                if config.get('text'):
                    self._builder.set_document_column(key)
                else:
                    if config.get('meta'):
                        dtype = config.get('dtype')
                        self._builder.add_metas(key, dtypes=dtype)
            button_output.clear_output()
            try:
                corpus = self._builder.build()
                # todo: remove this quick hack - for DH demo only.
                corpus = process(corpus, nlp=spacy.blank('en'), source='tweets')

                with button_output: print(f"{corpus_id.get('name')} added to registry.")
                self.update_registry(corpus_id.get('name'), corpus)
            except Exception as e:
                with button_output: print(f"Failed to build. {e}")
                return

        key_textbox.observe(_on_click_key_textbox, names='value')
        button.on_click(_on_click_build_corpus)

        return HBox([VBox(selection_widgets, layout=Layout(width='70%')),
                     VBox([key_textbox, button, button_output], layout=Layout(width='30%'))])

    ## Widget: Corpus Selector ##

    def corpus_registry(self):
        if self._corpus_selector is None:
            self._corpus_selector = GridBox([self._create_corpus_selector_table()],
                                            layout=Layout(grid_template_columns="repeat(2, 1fr)"))
        return self._corpus_selector

    def _update_corpus_selector(self, selected_key: str = None):
        """ Updates the Corpus Selector live with new registry. NO refresh required. """
        cs = self.corpus_registry()
        cs.children = (self._create_corpus_selector_table(selected_key),)
        # todo: triage/create new corpus table if it's full.

    def _create_corpus_selector_table(self, selected_key: str = None):
        hbox_registry_labels = HBox(self._corpus_selector_labels, layout=hbox_layout)
        rows = [self._create_corpus_selector_row(k) for k in self.REGISTRY.keys()]
        if selected_key:
            for r in rows:
                checkbox = r.children[0]
                checkbox.value = checkbox.description == selected_key
        return VBox([hbox_registry_labels] + rows)

    def _create_corpus_selector_row(self, corpus_id):
        checkbox = widgets.Checkbox(description=f"{corpus_id}", layout=_create_layout(**corpus_id_layout))
        checkbox.style = {'description_width': '0px'}
        checkbox.observe(self._observe_corpus_selector_checkbox, names='value')

        corpus = self.REGISTRY.get(corpus_id, None)
        if corpus is None: raise KeyError(f"Corpus ID: {corpus_id} does not exist.")
        size = len(corpus)
        parent_corpus = corpus.parent
        if parent_corpus is not None:
            parent = list(self.REGISTRY.keys())[list(self.REGISTRY.values()).index(parent_corpus)]
        else:
            parent = corpus_id
        if parent == corpus_id: parent = ''
        checkbox.add_class('corpus_id_focus_colour')  # todo: add this HTML to code
        return HBox([checkbox,
                     Label(str(size), layout=_create_layout(**size_layout)),
                     Label(parent, layout=_create_layout(**parent_layout))],
                    layout=hbox_layout)

    def _observe_corpus_selector_checkbox(self, event):
        if self._corpus_selector is None: return
        value, owner = event.get('new'), event.get('owner')
        if value:
            self._selected_corpus = self.REGISTRY.get(owner.description, None)  # todo: normalise this, then rm err
            if self._selected_corpus is None: raise RuntimeError("This should not happen.")
            # deselect everything else
            for vboxes in self._corpus_selector.children:
                for hboxes in vboxes.children:
                    for cb in hboxes.children:
                        if type(cb) == Checkbox:
                            cb.value = False if cb != owner else True

            # update corpus_slicer if exist.
            self._refresh_corpus_slicer()

    ## Widget: Corpus Slicer ##
    def corpus_slicer(self):
        # if self._selected_corpus is None:
        #     raise ValueError(f"No corpus selected. First run {self.corpus_registry.__name__}.")

        # reset dependencies
        self._corpus_slicer_dashboard = None  # corpus_slicer - for referencing
        self._corpus_slicer_operations = dict()  # corpus_slicer - stores all slicer operations.
        self._corpus_slicer_current_mask = None  # corpus_slicer - mask from all ops hist

        self._corpus_slicer = VBox([
            self.corpus_registry(),
            self._create_slice_operations_dashboard(), ],
            layout=Layout(width='100%'))
        return self._corpus_slicer

    def _refresh_corpus_slicer(self):
        if self._corpus_slicer is None: return
        self._corpus_slicer.children = (self._corpus_slicer.children[0], self._create_slice_operations_dashboard(),)

    def _create_meta_slicer(self):
        """ Creates the preview and slicing widget. """
        key_textbox = Text()

    def _create_slice_operations_dashboard(self):
        """ Creates a meta selector. """
        if self._selected_corpus is None: return Box()

        self._corpus_slicer_operations = dict()
        # meta
        options_meta = [id_ for id_ in self._selected_corpus.meta.keys()]
        label_meta = Label('Meta',
                           layout=_create_layout(**{'width': '98%', 'display': 'flex', 'justify_content': 'center'}))
        select_meta = Select(
            options=options_meta,
            value=options_meta[0] if len(options_meta) > 0 else None,
            disabled=False, layout=Layout(width='98%', height='100%')
        )

        # filter
        label_filter = Label('Filter By',
                             layout=_create_layout(**{'width': '98%', 'display': 'flex', 'justify_content': 'center'},
                                                   **debug_style))
        button_filter = Button(description='Add Operation', layout=Layout(width='98%', height='30px'))

        config_cache = dict()
        filter_value_cache = {m: self._create_slice_ops_selector(m, config_cache) for m in options_meta}

        # operation history
        label_hist = Label("Operations",
                           layout=_create_layout(**{'width': '98%', 'display': 'flex', 'justify_content': 'center'},
                                                 **debug_style))
        vbox_hist_cbs = VBox([],
                             layout=_create_layout(**{'width': '98%', 'display': 'flex', 'justify_content': 'center'},
                                                   **debug_style))
        # previews
        label_prevw = Label("Corpus Size",
                            layout=_create_layout(**{'width': '98%', 'display': 'flex', 'justify_content': 'center'},
                                                  **debug_style))

        # widgets.HTML(df.to_html(index=False, classes='table'))
        html_prevw = widgets.HTML(f'<h4>{len(self._selected_corpus)}</h4>',
                                  layout=_create_layout(**{'width': '98%', 'height': '40%'}))
        text_corpid_prevw = Text(placeholder='Corpus ID',
                                 layout=_create_layout(**{'width': '98%', 'height': '30%'}))
        button_prevw = Button(description="Slice",
                              disabled=True,
                              layout=_create_layout(**{'width': '98%', 'height': '30px'}))

        # vboxes & output
        vbox_meta = VBox([label_meta, select_meta],
                         layout=_create_layout(**{'width': '15%'}, **debug_style))
        vbox_filter = VBox([label_filter, filter_value_cache.get(select_meta.value, Box()), button_filter],
                           layout=_create_layout(**{'width': '40%'}, **debug_style, **center_style))
        vbox_prevw = VBox([label_prevw, html_prevw, text_corpid_prevw, button_prevw],
                          layout=_create_layout(**{'width': '15%'}, **debug_style))
        output = Output(layout=_create_layout(**{'width': '30%', 'height': '220px', 'overflow': 'scroll'}))

        vbox_hist = VBox([label_hist, vbox_hist_cbs],
                         layout=_create_layout(**{'width': '98%'}, **debug_style))

        # CALLBACKS
        def observe_select_meta(event):
            selected = event.get('new')
            vbox_filter.children = tuple([
                vbox_filter.children[0],
                filter_value_cache.get(selected),
                vbox_filter.children[2]
            ])

        select_meta.observe(observe_select_meta, names='value')

        def on_click_add(_):
            # print(f"Adding config: {config_cache}")
            selected = select_meta.value
            config = config_cache.get(selected)
            cb = self._create_ops_history_checkbox(selected, config, button_prevw)
            vbox_hist_cbs.children = (*vbox_hist_cbs.children, cb)

        button_filter.on_click(on_click_add)

        corp_id = dict()

        def observe_text_corpid(_):
            corp_id['text'] = text_corpid_prevw.value

        text_corpid_prevw.observe(observe_text_corpid, names='value')

        def on_click_slice(_):
            id_ = corp_id.get('text', None)
            if id_ is None:
                with output: print("You must enter a corpus ID.")
                return
            if self._corpus_slicer_current_mask is None: return
            try:
                self.update_registry(id_, self._selected_corpus.cloned(self._corpus_slicer_current_mask))
                selected_corpus_id = None
                for corpus_id, corpus in self.REGISTRY.items():
                    if corpus == self._selected_corpus:
                        selected_corpus_id = corpus_id
                        break
                self._update_corpus_selector(selected_corpus_id)
                if id_ not in self.REGISTRY.keys():
                    with output: print(f"Added {id_} to registry.")
            except KeyError as ke:
                with output: print(f"'{id_}' already exists. Please choose another key.")

        button_prevw.on_click(on_click_slice)
        self._corpus_slicer_dashboard = VBox([HBox([vbox_meta, vbox_filter, vbox_prevw, output],
                                                   layout=_create_layout(**{'width': '98%'})),
                                              vbox_hist], layout=_create_layout(**{'width': '98%'}))
        return self._corpus_slicer_dashboard

    def _create_slice_ops_selector(self, meta_id: str, config: dict):
        """ Creates a selector based on the meta selected. """
        meta = self._selected_corpus.meta.get(meta_id)
        if not isinstance(meta, SeriesMeta): raise NotImplementedError("Only supports SeriesMeta for now.")
        dtype = meta.series.dtype
        config[meta.id] = config.get(meta.id, dict())
        config = config.get(meta.id)
        if dtype == 'category':
            meta_value_selector = self._create_category_ops_selector(meta, config)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            meta_value_selector = self._create_datetime_ops_selector(meta, config)
        elif dtype in [int, pd.Int64Dtype()]:
            meta_value_selector = self._create_wholenumber_ops_selector(meta, config)
        elif dtype == float:
            meta_value_selector = self._create_decimal_ops_selector(meta, config)
        elif pd.api.types.is_string_dtype(dtype):
            meta_value_selector = self._create_text_ops_selector(meta, config)
        else:
            meta_value_selector = self._create_dummy_ops_selector(meta, config)
        return meta_value_selector

    def _create_dummy_ops_selector(self, meta: Meta, config):
        meta_value_selector = Checkbox(description=f"filter operation for {meta.id}",
                                       layout=_create_layout(**meta_value_selector_layout))

        def observe_update_config(event):
            config['event'] = event

        meta_value_selector.observe(observe_update_config, names='value')
        return meta_value_selector

    def _create_category_ops_selector(self, meta: SeriesMeta, config):
        options = sorted((str(item) for item in meta.series.unique().tolist()))
        widget = SelectMultiple(
            options=options,
            layout=_create_layout(**{'width': '98%'})
        )

        def update_category(event):
            config['item'] = widget.value

        update_category(None)  # initial set up
        widget.observe(update_category)
        return widget

    def _create_datetime_ops_selector(self, meta: SeriesMeta, config):
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
        slider = widgets.SelectionRangeSlider(options=options, index=index, layout={'width': '98%'})

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

    def _create_wholenumber_ops_selector(self, meta: SeriesMeta, config):
        """ DType: whole number; filter_by_item, filter_by_range """
        WIDGET_NUM = widgets.IntText

        ft_min = WIDGET_NUM(description='Min:', layout=Layout(width='98%'), value=meta.series.min())
        ft_max = WIDGET_NUM(description='Max:', layout=Layout(width='98%'), value=meta.series.max())
        vbox_range = VBox([ft_min, ft_max], layout=Layout(width='98%'))
        ft_num = WIDGET_NUM(description='Number:', layout=Layout(width='98%'))
        box_num = Box([ft_num], layout=Layout(width='98%'))

        vboxes = [
            ('Number', box_num),
            ('Min/Max', vbox_range)
        ]
        starter_idx = 0
        options = [vbox[starter_idx] for vbox in vboxes]
        toggle = widgets.ToggleButtons(
            options=options, value=options[starter_idx], layout=Layout(width='98%')
        )

        vbox = VBox([vboxes[starter_idx][1], toggle], layout=Layout(width='98%', height='65%'))

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

    def _create_decimal_ops_selector(self, meta: SeriesMeta, config):
        """ Dtype: decimal; filter_by_item, filter_by_range """
        WIDGET_NUM = widgets.FloatText

        ft_min = WIDGET_NUM(description='Min:', value=meta.series.min())
        ft_max = WIDGET_NUM(description='Max:', value=meta.series.max())
        vbox_range = VBox([ft_min, ft_max], layout=Layout(width='98%'))
        ft_num = WIDGET_NUM(description='Number:', layout=Layout(width='98%'))
        box_num = Box([ft_num], layout=Layout(width='98%'))

        vboxes = [
            ('Number', box_num),
            ('Min/Max', vbox_range)
        ]
        starter_idx = 0
        options = [vbox[starter_idx] for vbox in vboxes]
        toggle = widgets.ToggleButtons(
            options=options, value=options[starter_idx], layout=Layout(width='98%')
        )

        vbox = VBox([vboxes[starter_idx][1], toggle], layout=Layout(width='98%', height='65%'))

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

    def _create_text_ops_selector(self, meta: SeriesMeta, config):
        """ Dtype: text; filter_by_item, filter_by_regex """
        w_text = Text()
        w_toggle = widgets.ToggleButtons(
            options=['Text', 'Regex'],
            value='Text',
            description='',
            disabled=False,
            button_style='',
            tooltips=['Exact match', 'Enter a regex pattern'],
        )
        config['mode'] = w_toggle.value

        def observe_toggle(event):
            config['mode'] = event.get('new')

        def observe_text(event):
            config['text'] = event.get('new')

        w_toggle.observe(observe_toggle, names='value')
        w_text.observe(observe_text, names='value')
        return VBox([w_text, w_toggle], layout=Layout(width='98%'))

    def _create_ops_history_checkbox(self, selected, config, slice_button):
        cb = Checkbox(description=self._beautify_ops_checkbox_description(selected, config),
                      layout=_create_layout(**{'width': '98%'}))
        cb.style = {'description_width': '0px'}
        self._corpus_slicer_operations[cb] = (selected, deepcopy(config))

        def observe_cb(_):
            # calculates a mask given the checked boxes, outputs preview of corpus size.
            self._update_corpus_slicer_preview(html=f"<h4>Calculating...</h4>")
            self._corpus_slicer_current_mask = None
            mask = pd.Series([True] * len(self._selected_corpus), index=self._selected_corpus._df.index)
            for cb, (selected, config) in self._corpus_slicer_operations.items():
                if cb.value:
                    tmp_mask = self._filter_by_mask_triage(selected, config)
                    mask = mask & tmp_mask
            self._corpus_slicer_current_mask = mask
            num_docs = mask.sum()
            if num_docs > 0:
                html_text, disabled = f"<h4>{num_docs}</h4>", False
            else:
                html_text, disabled = f"<h4 style='color:red;'>{num_docs}</h4>", True

            self._update_corpus_slicer_preview(html=html_text)
            slice_button.disabled = disabled

        cb.observe(observe_cb, names='value')
        return cb

    def _update_corpus_slicer_preview(self, html: str):
        """ Updates the content of the corpus slicer preview."""
        if self._corpus_slicer_dashboard is None: return
        else:
            self._corpus_slicer_dashboard.children[0].children[2].children[1].value = html

    def _filter_by_mask_triage(self, selected, config: dict):
        meta = self._selected_corpus.meta.get_or_raise_err(selected)
        if 'start' in config.keys() and 'end' in config.keys():
            start, end = config.get('start'), config.get('end')
            mask = self._selected_corpus.slicer._filter_by_datetime_mask(meta, start, end, strftime=STRFORMAT)
        elif 'item' in config.keys():
            items = config.get('item')
            mask = self._selected_corpus.slicer._filter_by_item_mask(meta, items)
        elif 'mode' in config.keys():
            mode = config.get('mode').upper()
            if mode == 'NUMBER':
                mask = self._selected_corpus.slicer._filter_by_item_mask(meta, config.get('number'))
            elif mode == 'MIN/MAX':
                min_, max_ = config.get('range').get('min'), config.get('range').get('max')
                mask = self._selected_corpus.slicer._filter_by_range_mask(meta, min_, max_)
            elif mode == 'TEXT':
                mask = self._selected_corpus.slicer._filter_by_item_mask(meta, config.get('text'))
            elif mode == 'REGEX':
                mask = self._selected_corpus.slicer._filter_by_regex_mask(meta, config.get('text'), ignore_case=True)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return mask

    def _beautify_ops_checkbox_description(self, selected: str, config):
        prefix = f"[{selected.ljust(30)}] "
        if 'start' in config.keys() and 'end' in config.keys():
            start, end = config.get('start'), config.get('end')
            # text = f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"
            text = f"{start} - {end}"
        elif 'item' in config.keys():
            items = config.get('item')
            text = f"Items: {', '.join(items)}"
        elif 'mode' in config.keys():
            mode = config.get('mode').upper()
            if mode == 'NUMBER':
                text = f"Number: {config.get('number')}"
            elif mode == 'MIN/MAX':
                min_, max_ = config.get('range').get('min'), config.get('range').get('max')
                text = f"Min: {min_}\tMax: {max_}"
            elif mode == 'TEXT':
                text = f"Text: {config.get('text')}"
            elif mode == 'REGEX':
                text = f"Regex: {config.get('text')}"
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return prefix + text

    def reset(self):
        self._corpus_selector = None

    def __getitem__(self, corpus_id: str):
        return self.REGISTRY.get(corpus_id, None)

    def __setitem__(self, corpus_id: str, corpus: Corpus):
        if not isinstance(corpus, Corpus):
            raise ValueError(f"{corpus} must be an instance of {Corpus.__class__.__name__}")
        # self.REGISTRY[corpus_id] = corpus
        # self._update_corpus_selector()
        self.update_registry(corpus_id, corpus)

    def __delitem__(self, corpus_id: str):
        del self.REGISTRY[corpus_id]
        self._update_corpus_selector()


######################################################################################################
hbox_layout = Layout(display='inline-flex',
                     flex_flow='row',  # short for: flex-direction flex-wrap
                     align_items='stretch',  # assigns align-self to all direct children, affects height,
                     width='100%')
# 'flex' short for: flex-grow, flex-shrink, flex-basis
debug_style = {'border': '0px solid blue'}
center_style = {'display': 'flex', 'justify_content': 'center'}
corpus_id_layout = {'width': '40%'}
size_layout = {'width': '20%'}
parent_layout = {'width': '40%'}

# corpus builder
f_selector_layout = {'width': '98%', 'height': '100%'}
f_uploader_layout = {'width': '98%', 'height': '50px'}
box_df_layout = {'width': '100%', 'height': '100%'}

# corpus slicer
meta_value_selector_layout = {'width': '98%', 'height': '60%'}


def _create_layout(**kwargs):
    return Layout(**kwargs)
