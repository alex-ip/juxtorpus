"""
Handlers
"""
from abc import ABCMeta, abstractmethod


class InputHandler(object, metaclass=ABCMeta):
    @abstractmethod
    def handle_file(self):
        raise NotImplementedError()

    @abstractmethod
    def handle_line(self):
        raise NotImplementedError()


from enum import Enum


class SizeHandler(InputHandler):
    class Mode(Enum):
        Warn = "Warns the user when a file exceeds a certain size."
        Error = "Raises an error when a file exceeds a certain size."

    def __init__(self, mode: Mode, size_in_bytes: int):
        pass

    def handle_line(self):
        pass

    def handle_file(self):
        pass


class LangHandler(InputHandler):
    class Mode(Enum):
        Warn = "Warns the user when a "