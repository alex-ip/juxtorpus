""" App

Registry holds all the corpus and sliced subcorpus in memory. Allowing on the fly access.
"""
import pandas as pd
from ipywidgets import Layout, Label, HBox, VBox, GridBox, Checkbox, SelectMultiple, Box, Button
import ipywidgets as widgets
from pathlib import Path
import math

from juxtorpus.corpus import Corpus, CorpusBuilder
from juxtorpus.viz.widgets import FileUploadWidget


def format_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class App(object):
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
        self._corpus_selector = None
        self._corpus_builder = None

    ## Corpus Registry ##
    def update_registry(self, corpus_id, corpus):
        if corpus_id in self.REGISTRY.keys():
            res = input(f"{corpus_id} already exists. Overwrite (y/n)? ")
            if res != 'y': return
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

        hbox_corpus_builder = self._create_corpus_builder()

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
            box_df.children = (widgets.HTML(df.to_html(index=False, classes='table table-stripped')),)

        f_selector.observe(_observe_file_selected, names='value')
        button_confirm = Button(description='Confirm',
                                layout=Layout(width='20%', height='50px'))
        hbox_uploader = HBox([VBox([f_selector, fuw._uploader], layout=Layout(width='50%', height='200px')),
                              VBox([box_df, button_confirm], layout=Layout(width='50%', height='200px'))],
                             layout=Layout(width='100%', height='100%'))

        vbox = VBox([hbox_uploader, hbox_corpus_builder])

        def on_click_confirm(_):
            selected_files = [d.get('path') for d in self._files.values() if d.get('selected')]
            if len(selected_files) < 0:
                print("No files selected.")
                return
            self._builder = CorpusBuilder(selected_files)
            # hbox_corpus_builder.children = tuple((Label(p.name) for p in self._builder.paths))
            hbox_corpus_builder = self._create_corpus_builder()
            vbox.children = (vbox.children[0], hbox_corpus_builder)

        button_confirm.on_click(on_click_confirm)
        return vbox

    # Widget: Corpus Builder
    def _create_corpus_builder(self):
        if self._builder is None: return VBox()

        def create_text_meta_dtype_row(id_: str, text_checked: bool, meta_checked: bool, dtypes: list[str]):
            label = widgets.Label(f"{id_}", layout=widgets.Layout(width='30%'))
            t_checkbox = widgets.Checkbox(value=text_checked, layout=widgets.Layout(width='15%'))
            t_checkbox.style.description_width = '0px'
            m_checkbox = widgets.Checkbox(value=meta_checked, layout=widgets.Layout(width='15%'))
            m_checkbox.style.description_width = '0px'
            m_checkbox.observe(lambda e: print(e), names='value')       # todo: toggle text and meta
            dtype_dd = widgets.Dropdown(options=dtypes, value=dtypes[0], disabled=False,
                                        layout=widgets.Layout(width='100px'))
            return widgets.HBox([label, t_checkbox, m_checkbox, dtype_dd])

        top_labels = [('id', '30%'), ('text', '15%'), ('meta', '15%'), ('dtype', '30%')]
        selection_top_labels = widgets.HBox(
            list(map(lambda ls: widgets.HTML(f"<b>{ls[0]}</b>", layout=widgets.Layout(width=ls[1])), top_labels)))

        dtypes = sorted(list(CorpusBuilder.allowed_dtypes))
        columns = self._builder.columns
        selection_widgets = [selection_top_labels]
        selection_widgets.extend([create_text_meta_dtype_row(col, False, True, dtypes) for col in columns])
        # return HBox([Label("hello")], layout=Layout(width='100%', height='100%'))
        return VBox(selection_widgets, layout=Layout(width='100%'))

    ## Widget: Corpus Selector ##
    def corpus_selector(self):
        if self._corpus_selector is None:
            self._corpus_selector = GridBox([self._create_corpus_selector_table()],
                                            layout=Layout(grid_template_columns="repeat(2, 1fr)"))
        return self._corpus_selector

    def _update_corpus_selector(self):
        """ Updates the Corpus Selector live with new registry. NO refresh required. """
        cs = self.corpus_selector()
        cs.children = (self._create_corpus_selector_table(),)
        # todo: triage/create new corpus table if it's full.

    def _create_corpus_selector_table(self):
        hbox_registry_labels = HBox(self._corpus_selector_labels, layout=hbox_layout)
        return VBox([hbox_registry_labels] + [self._create_corpus_selector_row(k) for k in self.REGISTRY.keys()])

    def _create_corpus_selector_row(self, corpus_id):
        checkbox = widgets.Checkbox(description=f"{corpus_id}")
        checkbox.style = {'description_width': '0px'}
        checkbox.observe(self._observe_corpus_selector_checkbox)

        corpus = self.REGISTRY.get(corpus_id, None)
        if corpus is None: raise KeyError(f"Corpus ID: {corpus_id} does not exist.")
        size = len(corpus)
        parent_corpus = corpus.find_root()
        parent = list(self.REGISTRY.keys())[list(self.REGISTRY.values()).index(parent_corpus)]
        if parent == corpus_id: parent = ''
        checkbox.layout = _create_layout(**corpus_id_layout)
        checkbox.add_class('corpus_id_focus_colour')  # todo: add this HTML to code
        return HBox([checkbox,
                     Label(str(size), layout=_create_layout(**size_layout)),
                     Label(parent, layout=_create_layout(**parent_layout))],
                    layout=hbox_layout)

    def _observe_corpus_selector_checkbox(self, event):
        value, owner = event.get('new'), event.get('owner')
        if value:
            # deselect everything else
            for child in self.corpus_selector():
                for cb in child.children:
                    if type(cb) == Checkbox:
                        cb.value = False if cb != owner else True
            self._selected_corpus = self.REGISTRY.get(owner.description, None)  # todo: normalise this, then rm err
            if self._selected_corpus is None: raise RuntimeError("This should not happen.")

    ## Widget: Corpus Slicer ##
    def corpus_slicer(self):
        pass

    def reset(self):
        self._corpus_selector = None


######################################################################################################
hbox_layout = Layout(display='inline-flex',
                     flex_flow='row',  # short for: flex-direction flex-wrap
                     align_items='stretch',  # assigns align-self to all direct children, affects height,
                     width='100%')
# 'flex' short for: flex-grow, flex-shrink, flex-basis
debug_style = {'border': '0px solid blue'}
center_style = {'display': 'flex', 'justify_content': 'center'}
corpus_id_layout = {'width': '40%', **debug_style}
size_layout = {'width': '20%', **debug_style}
parent_layout = {'width': '40%', **debug_style}

# corpus builder
f_selector_layout = {'width': '98%', 'height': '100%'}
f_uploader_layout = {'width': '98%', 'height': '50px'}
box_df_layout = {'width': '100%', 'height': '100%'}


def _create_layout(**kwargs):
    return Layout(**kwargs)
