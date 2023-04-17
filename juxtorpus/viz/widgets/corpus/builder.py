from typing import Callable
from ipywidgets import (SelectMultiple, Layout, Box, HTML, Button, HBox, VBox,
                        Output, Text, Label, Dropdown, Checkbox)
import pandas as pd
import math

from juxtorpus.viz import Widget
from juxtorpus.viz.widgets import FileUploadWidget

f_selector_layout = {'width': '98%', 'height': '100%'}
f_uploader_layout = {'width': '98%', 'height': '50px'}
box_df_layout = {'width': '100%', 'height': '100%'}


def format_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class CorpusBuilderWidget(Widget):
    def __init__(self, builder: 'CorpusBuilder'):
        self.builder = builder
        self._on_build_callback = lambda corpus: None

    def widget(self):
        top_labels = [('id', '30%'), ('document', '15%'), ('meta', '15%'), ('data type', '30%')]
        top_labels = HBox(
            list(map(lambda ls: HTML(f"<b>{ls[0]}</b>", layout=Layout(width=ls[1])), top_labels))
        )

        checkbox_configs = {col: {'text': False, 'meta': True} for col in self.builder.columns}
        panel = [top_labels] + [self._create_checkbox(col, config.get('text'), config.get('meta'), config)
                                for col, config in checkbox_configs.items()]

        key_textbox = Text(description='Name:', placeholder='Corpus Name (randomly generates if not supplied)')
        corpus_name = dict(name=None)

        button = Button(description='Build')
        button_output = Output(layout=Layout(overflow='scroll hidden'))

        def _on_click_key_textbox(event):
            corpus_name.update(name=event.get('new'))

        def _on_click_build_corpus(_):
            if button.description == 'Done.':
                button.description = 'Build'
                return
            for key, config in checkbox_configs.items():
                if config.get('text'):
                    self.builder.set_document_column(key)
                else:
                    if config.get('meta'):
                        dtype = config.get('dtype')
                        self.builder.add_metas(key, dtypes=dtype)
            button_output.clear_output()
            try:
                button.description = "Building..."
                corpus = self.builder.build()
                if self._on_build_callback is not None:
                    self._on_build_callback(corpus)
                button.description = "Done."
            except Exception as e:
                with button_output: print(f"Failed to build. {e}")
                button.description = 'Build'
                return

        key_textbox.observe(_on_click_key_textbox, names='value')
        button.on_click(_on_click_build_corpus)

        return HBox([VBox(panel, layout=Layout(width='70%')),
                     VBox([key_textbox, button, button_output], layout=Layout(width='30%'))])

    def _create_checkbox(self, id_: str, text_checked: bool, meta_checked: bool, config: dict):
        label = Label(f"{id_}", layout=Layout(width='30%'))
        t_checkbox = Checkbox(value=text_checked, layout=Layout(width='15%'))
        t_checkbox.style.description_width = '0px'

        m_checkbox = Checkbox(value=meta_checked, layout=Layout(width='15%'))
        m_checkbox.style.description_width = '0px'

        dtypes = sorted([k for k in self.WIDGET_DTYPES_MAP.keys()])
        dtype_dd = Dropdown(options=dtypes, value=dtypes[0], disabled=False,
                            layout=Layout(width='100px'))

        # dtype_dd.observe
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
            config['dtype'] = self.WIDGET_DTYPES_MAP.get(event.get('new'))

        t_checkbox.observe(_toggle_checkbox, names='value')
        m_checkbox.observe(_toggle_checkbox, names='value')  # todo: toggle text and meta
        dtype_dd.observe(_update_dtype, names='value')
        return HBox([label, t_checkbox, m_checkbox, dtype_dd])

    WIDGET_DTYPES_MAP = {
        'auto': None,
        'decimal': 'float',
        'whole number': 'Int64',
        'text': 'str',
        'datetime': 'datetime',
        'category': 'category'
    }

    def set_callback(self, callback: Callable):
        self._on_build_callback = callback


class CorpusBuilderFileUploadWidget(Widget):
    """ CorpusBuilderWidget
    Workflow:
    1. upload file(s)
    2. invoke corpus builder widget.
    """

    DTYPES_MAP = {
        'auto': None,
        'decimal': 'float',
        'whole number': 'Int64',
        'text': 'str',
        'datetime': 'datetime',
        'category': 'category'
    }

    def __init__(self):
        self._files = dict()
        self._builder = None
        self._widget = self._create_file_uploader_entrypoint()

        # callbacks
        self._on_build_callback = lambda corpus, output: None

    def widget(self):
        return self._widget

    def _create_file_uploader_entrypoint(self):
        """ Creates a selectable FileUploaderWidget. This is the entrypoint. """
        f_selector = SelectMultiple(layout=Layout(**f_selector_layout))
        fuw = FileUploadWidget()
        fuw._uploader.layout = Layout(**f_uploader_layout)

        def _callback_fileupload_to_fileselector(fuw, added):
            self._files = {p.name: {'path': p} for p in fuw.uploaded()}
            f_selector.options = [name for name in self._files.keys()]

        fuw.set_callback(_callback_fileupload_to_fileselector)

        box_file_stats = Box(layout=Layout(**box_df_layout))

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
            box_file_stats.children = (HTML(df.to_html(index=False, classes='table')),)

        f_selector.observe(_observe_file_selected, names='value')
        button_confirm = Button(description='Confirm',
                                layout=Layout(width='20%', height='50px'))
        hbox_uploader = HBox([VBox([f_selector, fuw._uploader], layout=Layout(width='50%', height='200px')),
                              VBox([box_file_stats, button_confirm], layout=Layout(width='50%', height='200px'))],
                             layout=Layout(width='100%', height='100%'))

        vbox = VBox([hbox_uploader, Box()])

        def on_click_confirm(_):
            from juxtorpus.corpus import CorpusBuilder
            selected_files = [d.get('path') for d in self._files.values() if d.get('selected')]
            if len(selected_files) <= 0: return
            builder = CorpusBuilder(selected_files)
            builder.set_callback(self._on_build_callback)
            vbox.children = (vbox.children[0], builder.widget())

        button_confirm.on_click(on_click_confirm)
        return vbox

    def set_callback(self, callback: Callable):
        """ Callbacks that takes the built corpus as the argument. """
        self._on_build_callback = callback
