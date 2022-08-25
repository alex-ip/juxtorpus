"""
SpaCy Custom Components

This package contains subclasses of Component. These subclasses are custom spaCy components that you create and
can be added to the nlp pipeline.

Documentations:
https://spacy.io/usage/processing-pipelines#custom-components
https://explosion.ai/blog/spacy-v2-pipelines-extensions
"""
import abc
from typing import Dict, Union
from abc import ABCMeta, abstractmethod

from spacy.tokens.doc import Doc


class Component(metaclass=ABCMeta):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, doc: Doc) -> Doc:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def cache(cls) -> bool:
        """ Whether the component is cached. Override this method to determine if it is. """
        raise NotImplementedError()


class ComponentImpl(Component):

    def __init__(self, name: str):
        super(ComponentImpl, self).__init__(name)

    def __call__(self, doc: Doc):
        print(f"Processing doc: {doc}...")
        return doc

    @classmethod
    def cache(cls) -> bool:
        return False


# NOTE: Code below will need to be abstracted to factories o allow for different inits. No need for this yet I guess.
# caches for pipeline components with expensive (time or memory) initialisation.
# key = class name (must be a subclass of Component)
_cache: Dict[str, Component] = dict()


def get_component(name: str) -> Union[Component, None]:
    """ Returns a pipeline component either from cache or brand new. """
    if _cache.get(name, None):

        # perf: number of subclasses expected to be small so it shouldn't be a concern for the time being.
        subcls: Union[abc.ABCMeta, None] = None
        for _subcls in Component.__subclasses__():
            if name == _subcls.__name__:
                subcls = _subcls
                break
        if subcls is None: return None
        if getattr(subcls, 'cache')(subcls):
            _cache[subcls.__name__] = subcls()  # TODO: this won't allow for different inits!
            return _cache.get(subcls.__name__)
        else:
            return subcls()


if __name__ == '__main__':
    ci = ComponentImpl(name='')
    print(Component.__subclasses__())
    print([cls.__name__ for cls in Component.__subclasses__()])
