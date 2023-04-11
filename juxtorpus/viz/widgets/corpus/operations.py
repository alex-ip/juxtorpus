from typing import Callable, Optional, Union
import pandas as pd
from ipywidgets import *
from juxtorpus.interfaces import Container
from juxtorpus.viz import Widget
from juxtorpus.corpus.operation import Operation
from juxtorpus.corpus.operations import Operations
from juxtorpus.viz.style.ipyw import *


class OperationsWidget(Widget):
    def __init__(self, corpus: 'Corpus', operations: Operations):
        self.corpus = corpus
        self._sliced = None  # holder for sliced corpus.
        self.ops = operations

        # internal mutating states
        self._checkbox_to_op = dict()  # note: this must be an ordered dict.
        self._preview = self._sliced_preview(' ')

        # callbacks
        self._on_slice_callback = lambda sliced: sliced
        self._btn_slice = self._slice_button()

    @property
    def sliced(self):
        return self._sliced

    def widget(self):
        """ Returns a checkbox table with a Subcorpus Preview and Slice button next to it."""
        ops_table = self._ops_table()
        vbox_slice = VBox([self._preview, self._btn_slice],
                          layout=Layout(height='100%', **no_horizontal_scroll))
        return HBox([ops_table, vbox_slice], layout=Layout(**no_horizontal_scroll))

    def _populate_checkbox_to_op_map(self):
        for op in self.ops:
            if op not in self._checkbox_to_op.values():
                self._checkbox_to_op[self._ops_row(op)] = op

    def _ops_table(self):
        """ Returns a table of toggleable operations. """
        self._populate_checkbox_to_op_map()
        checkboxes = [checkbox for checkbox in self._checkbox_to_op.keys()]
        for cb in checkboxes: cb.value = True
        return VBox(checkboxes,
                    layout=Layout(height='100%', **no_horizontal_scroll))

    def _ops_row(self, op: Operation):
        """ Returns a toggleable checkbox associated with each operation. """
        formatted = self._op_description(op)
        checkbox = Checkbox(description=formatted, layout=Layout(**no_horizontal_scroll, **debug_style))
        checkbox.style = {'description_width': '0px'}

        def _observe(event):
            self._update_sliced_preview()

        checkbox.observe(_observe, names='value')
        return checkbox

    def _slice_button(self):
        """ Slice button should """
        button = Button(description='Slice', disabled=True,
                        layout=Layout(height='30px', **no_horizontal_scroll))

        def _on_click(event):
            subcorpus = self.corpus
            for cb, op in self._checkbox_to_op.items():
                if cb.value: subcorpus = op.apply(subcorpus)
            self._sliced = subcorpus
            self._on_slice_callback(subcorpus)

        button.on_click(_on_click)
        return button

    def _sliced_preview(self, text: str):
        """ Returns the preview box of sliced corpus before slicing. """
        return HTML(self._preview_text(text),
                    placeholder='Corpus Size: ',
                    layout=Layout(height='100%', **no_horizontal_scroll))

    def _update_sliced_preview(self):
        self._preview.value = self._preview_text("Calculating...")
        mask = self.mask()
        subcorpus_size = mask.sum()
        self._preview.value = self._preview_text(str(subcorpus_size))
        self._btn_slice.disabled = not subcorpus_size > 0

    def mask(self):
        toggled_indices = [i for i, cb in enumerate(self._checkbox_to_op.keys()) if cb.value]
        ops = [self.ops.get(idx) for idx in toggled_indices]
        if len(ops) <= 0:
            return pd.Series([True] * len(self.corpus))
        else:
            mask = ops[0].mask()
            for op in ops[1:]: mask = mask & op.mask()
            return mask

    @staticmethod
    def _op_description(op: Operation) -> str:
        """ Format Operation into a checkbox description """
        return f"{op.__class__.__name__}"  # todo: this is a placeholder.
        # NOTE: a quick note - i think it checkbox descriptions can't have angular brackets.

    @staticmethod
    def _preview_text(text: str):
        return f"<h4>Corpus Size: {text}</h4>"

    def set_callback(self, callback: Callable):
        """ Sets on slice callback. Receives one argument: sliced corpus."""
        self._on_slice_callback = callback
