import pandas as pd
from copy import deepcopy
from .base import Meta
from .series import SeriesMeta
from .spacy_doc import DocMeta

""" Metadata from spaCy docs can only be derived metadata. """


class MetaRegistry(dict):
    def __init__(self, *args, **kwargs):
        super(MetaRegistry, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, Meta): raise ValueError(f"MetaRegistry only holds {Meta.__name__} objects.")
        super(MetaRegistry, self).__setitem__(key, value)

    def summary(self):
        """ Returns a summary of the metadata information. """
        infos = (meta.summary() for meta in self.values())
        df = pd.concat(infos, axis=0).fillna('')

        return df.T

    def copy(self):
        return deepcopy(self)

    def get_or_raise_err(self, id_: str):
        meta = self.get(id_, None)
        if meta is None: raise LookupError(f"{id_} does not exist.")
        return meta
