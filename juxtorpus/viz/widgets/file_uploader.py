from ipywidgets import FileUpload, Output, VBox, widgets
from IPython.display import display
from typing import Union
import pathlib
import os

from juxtorpus.viz import Viz


class FileUploadWidget(Viz):
    DESCRIPTION = "Upload your files here.\n({})"
    ERR_FAILED_UPLOAD = "File upload unsuccessful. Please try again!."

    default_accepted_extensions = ['.txt', '.csv', '.xlsx', '.zip']

    def __init__(self, upload_dir: Union[str, pathlib.Path], accept_extensions: list[str] = None):
        if accept_extensions is None:
            accept_extensions = self.default_accepted_extensions

        # TODO: probably better to use a temp directory here. (tmp will take care of deleting files)
        self.upload_dir = upload_dir if isinstance(upload_dir, pathlib.Path) else pathlib.Path(upload_dir)
        self._init_upload_dir()

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

        self._persisted_paths = list()

    def uploaded(self):
        return self._persisted_paths

    def render(self):
        return display(VBox([self._uploader, self._output]))

    def _on_upload(self, change):
        with self._output:
            new_files = change.get('new').keys()
            for fname in new_files:
                try:
                    path = self.upload_dir.joinpath(pathlib.Path(fname))
                    self._write_to_disk(fname=path, content=change.get('new').get(fname).get('content'))
                    print(f"++ Successfully wrote {fname} to disk...")
                    self._persisted_paths.append(path.absolute())
                except IOError as ioe:
                    print(f"-- Failed to write {fname} to disk... {ioe}")
                except Exception as e:
                    print(f"-- Failed to write {fname} to disk... {e}")

    def _write_to_disk(self, fname: Union[str, pathlib.Path], content: bytes):
        with open(fname, 'wb') as fh:
            fh.write(content)

    def _init_upload_dir(self):
        try:
            if not self.upload_dir.exists():
                os.mkdir(self.upload_dir)
        except Exception:
            raise IOError("Unable to create upload directory. Use another one?")
