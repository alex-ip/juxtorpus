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
        self._current_subcorpus_mask = None

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
            self._btn_slice.description = "Slice"
            self._update_current_mask()
            self._update_sliced_preview()
            at_least_one_checked = sum(cb.value for cb in self._checkbox_to_op.keys()) > 0
            self._btn_slice.disabled = not self._current_subcorpus_mask.sum() > 0 or not at_least_one_checked

        checkbox.observe(_observe, names='value')
        return checkbox

    def _slice_button(self):
        """ Slice button should """
        button = Button(description='Slice', disabled=True,
                        layout=Layout(height='30px', **no_horizontal_scroll))

        def _on_click(event):
            # NOTE: this depends on observer on operations checkboxes, else mask won't be updated.
            subcorpus = self.corpus.cloned(self._current_subcorpus_mask)
            self._sliced = subcorpus
            self._on_slice_callback(subcorpus)
            button.description = "Done."

        button.on_click(_on_click)
        return button

    def _sliced_preview(self, text: str):
        """ Returns the preview box of sliced corpus before slicing. """
        return HTML(self._preview_text(text),
                    placeholder='Corpus Size: ',
                    layout=Layout(height='100%', **no_horizontal_scroll))

    def _update_current_mask(self):
        self._current_subcorpus_mask = self.mask()

    def _update_sliced_preview(self):
        self._preview.value = self._preview_text("Calculating...")
        if self._current_subcorpus_mask is None: self._update_current_mask()
        subcorpus_size = self._current_subcorpus_mask.sum()
        self._preview.value = self._preview_text(str(subcorpus_size))

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
        return f"{str(op).replace('<', '').replace('>', '')}"
        # NOTE: a quick note - checkbox descriptions can't have angular brackets.

    @staticmethod
    def _preview_text(text: str):
        return f"<h4>Corpus Size: {text}</h4>"

    def set_callback(self, callback: Callable):
        """ Sets on slice callback. Receives one argument: sliced corpus."""
        self._on_slice_callback = callback
