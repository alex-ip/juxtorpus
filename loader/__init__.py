""" Loader class

21.08.22
This class serves as the first loader and is expected to be refactored as more loading procedures are supported.
It will ideally be responsible for the following functions...

1. unified interface to load from various inputs (e.g. upload widget, csv, parquets)
2. perform automatic dtype recognition for memory enhancements in pandas dataframes.
"""
import io, pathlib
from typing import Union, List
from collections.abc import Generator, Iterable

from handlers import InputHandler


# factory method to produce inputs
def get_input(input_: Union[io.IOBase, pathlib.PosixPath, Generator[pathlib.PosixPath], str]) -> 'Input':
    if isinstance(input_, Generator):
        return Inputs(input_)
    else:
        return Input(input_)


class Input(object):
    """ Input is an information provider of various input formats such as files (csv, txt).

    It follows the command and iterator pattern. Command to provide the information that a set of
    handlers may require. Iterator to provide granular 'per line' access.
    """

    def __init__(self, input_: Union[io.IOBase, Generator[pathlib.PosixPath], str]):
        if isinstance(input_, str):
            _path = pathlib.Path(input_)
            if _path.exists():
                print("Check if its file or directory, then construct the input object.")
                self._path = _path
                # append input streams
            else:
                input_ = io.StringIO(input_)

        self._input_stream = input_

    def __iter__(self):
        pass


class Inputs(Input):
    """ Collection of input object. It is itself an Input and exposes the same behaviours as is required by
     combining all individual input objects.
     """
    pass


from juxtorpus.viz import Viz
from IPython import display


class FileUploadWidget(Input, Viz):
    def render(self):
        raise NotImplementedError("Run IPython.display() on the file upload widget")


class Loader(object):
    """ The Loader performs a series of checks from Input """

    def __init__(self, input_: Input, handlers: List[InputHandler]):
        self._input = input_
        self.nrows = None

    def set_nrows(self, nrows: int):
        self.nrows = nrows

    def load(self) -> io.IOBase:
        pass

    # todo: context manager -> with Loader(input) as h: print(h)
