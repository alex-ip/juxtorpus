import spacy
from typing import Union
from juxtorpus.components import *

model: str = 'en_core_web_sm'
nlp: Union[spacy.Language, None] = spacy.load(model)


def reload_spacy(model_: str, clear_mem: bool):
    global model, nlp
    model = model_
    nlp = spacy.load(model)
    if clear_mem:
        import gc
        gc.collect()
