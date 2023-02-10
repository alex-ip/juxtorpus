""" App

Registry holds all the corpus and sliced subcorpus in memory. Allowing on the fly access.
"""
from ipywidgets import Layout, Label, HBox, VBox, GridBox, Checkbox
import ipywidgets as widgets

# TODO: temporary solution for registry
REGISTRY = dict()


def update_registry(corpus_id, corpus):
    if corpus_id in REGISTRY.keys():
        res = input(f"{corpus_id} already exists. Overwrite (y/n)? ")
        if res != 'y': return
    REGISTRY[corpus_id] = corpus


def corpus_slicer():
    pass


class App(object):
    def __init__(self):
        self._CORPUS_SELECTOR_MAX_NUM_CORPUS_PER_COLUMN = 20
        self._corpus_selector_labels = [
            Label("Corpus ID", layout=_create_layout(**corpus_id_layout, **center_style)),
            Label("Size", layout=_create_layout(**size_layout, **center_style)),
            Label("Parent", layout=_create_layout(**parent_layout, **center_style))
        ]

        # widgets
        self._corpus_selector = None

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
        return VBox([hbox_registry_labels] + [self._create_corpus_selector_row(k) for k in REGISTRY.keys()])

    def _create_corpus_selector_row(self, corpus_id):
        checkbox = widgets.Checkbox(description=f"{corpus_id}")
        checkbox.style = {'description_width': '0px'}

        corpus = REGISTRY.get(corpus_id, None)
        if corpus is None: raise KeyError(f"Corpus ID: {corpus_id} does not exist.")
        size = len(corpus)
        parent_corpus = corpus.find_root()
        parent = list(REGISTRY.keys())[list(REGISTRY.values()).index(parent_corpus)]
        if parent == corpus_id: parent = ''
        checkbox.layout = _create_layout(**corpus_id_layout)
        checkbox.add_class('corpus_id_focus_colour')
        return HBox([checkbox,
                     Label(str(size), layout=_create_layout(**size_layout)),
                     Label(parent, layout=_create_layout(**parent_layout))],
                    layout=hbox_layout)

    ## Widget: Corpus Slicer ##


    def reset(self):
        self._corpus_selector = None


######################################################################################################
# TODO: move registry widget code here.

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


def _create_layout(**kwargs):
    return Layout(**kwargs)
