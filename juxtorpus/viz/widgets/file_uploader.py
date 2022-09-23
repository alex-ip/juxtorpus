import tempfile

from ipywidgets import FileUpload, Output, VBox, widgets
from IPython.display import display
from typing import Union
import pathlib
import os

from juxtorpus.viz import Viz
from juxtorpus.utils import DeduplicatedDirectory

"""
NOTE: File size limit using jupyter notebooks.

start jupyter notebook with:
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
"""


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
                content = change.get('new').get(fname).get('content')
                if fname.endswith('.zip'):
                    self._add_zip(content, fname)
                else:
                    self._add_file(content, fname)

    def _add_zip(self, content, fname):
        try:
            print(f"++ Writing {fname} to disk...", end='')
            tmp_zip_dir = pathlib.Path(tempfile.mkdtemp())
            tmp_zip_file = tmp_zip_dir.joinpath(fname)
            with open(tmp_zip_file, 'wb') as fh:
                fh.write(content)
            self._dir.add_zip(tmp_zip_file)
            print("Success.")
        except Exception as e:
            print(f"Failed. Reason: {e}")

    def _add_file(self, content, fname):
        try:
            print(f"++ Writing {fname} to disk...", end='')
            self._dir.add_content(content, fname)
            print("Success.")
        except ValueError as e:
            print(f"Failed. Reason: {e}")
