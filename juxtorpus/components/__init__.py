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
from spacy.language import Language


class Component(metaclass=ABCMeta):
    def __init__(self, nlp: Language, name: str):
        self._nlp = nlp
        self._name = name

    @abstractmethod
    def __call__(self, doc: Doc) -> Doc:
        raise NotImplementedError()


class ComponentImpl(Component):

    def __init__(self, nlp: Language, name: str):
        super(ComponentImpl, self).__init__(nlp, name)
        print(name)

    def __call__(self, doc: Doc):
        print(f"{self.__call__}: Processing doc: {doc}...")
        return doc


@Language.component("stateless_custom_component")
def stateless_custom_component(doc: Doc):
    print("I don't do anything but I demonstrate a stateless component implementation.")
    return doc


# EACH FACTORY wrapper represents an INSTANCE of the stateful component.
@Language.factory(name='stateful_custom_component', default_config={"a_setting": True})
def stateful_custom_component(nlp: Language, name: str, a_setting):
    print(name)  # this is the instance name.
    return ComponentImpl(nlp, name)


# NOTE: Code below will need to be abstracted to factories o allow for different inits. No need for this yet I guess.
# caches for pipeline components with expensive (time or memory) initialisation.
# key = class name (must be a subclass of Component)


if __name__ == '__main__':
    import spacy

    nlp = spacy.blank('en')
    nlp.add_pipe("stateful_custom_component", config={"a_setting": False})  # Notice default setting is True.
    nlp("This is a sentence.")
