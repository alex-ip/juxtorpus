from ipywidgets import FileUpload, Output, VBox, widgets
from IPython.display import display
from typing import Union
import pathlib
import os

from juxtorpus.viz import Viz
from juxtorpus.utils import DeduplicatedDirectory


class FileUploadWidget(Viz):
    DESCRIPTION = "Upload your files here.\n({})"
    ERR_FAILED_UPLOAD = "File upload unsuccessful. Please try again!."

    default_accepted_extensions = ['.txt', '.csv', '.xlsx', '.zip']

    def __init__(self, accept_extensions: list[str] = None):
        if accept_extensions is None:
            accept_extensions = self.default_accepted_extensions

        self._dir = DeduplicatedDirectory()

        self._uploader = FileUpload(
            description=self.DESCRIPTION.format(' '.join(accept_extensions)),
            accept=', '.join(accept_extensions),
            multiple=True,  # True to accept multiple files
            error=self.ERR_FAILED_UPLOAD,
            layout=widgets.Layout(width='320px')
        )
        # required for ipython renders
        self._output = Output()
        self._uploader.observe(self._on_upload, names=['value'])  # 'value' for when any file is uploaded.

    def uploaded(self):
        return self._dir.files()

    def render(self):
        return display(VBox([self._uploader, self._output]))

    def _on_upload(self, change):
        with self._output:
            new_files = change.get('new').keys()
            for fname in new_files:
                try:
                    self._dir.add_content(change.get('new').get(fname).get('content'), fname)
                    print(f"++ Successfully wrote {fname} to disk...")
                except ValueError as e:
                    print(f"-- Failed to write {fname} to disk... {e}")
