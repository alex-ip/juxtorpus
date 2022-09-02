import spacy
from typing import Union
from juxtorpus.components import *

model: str = 'en_core_web_sm'
nlp: Union[spacy.Language, None] = spacy.load(model)
out_of_the_box_components = list(nlp.meta.get('components'))


def reload_spacy(model_: str, clear_mem: bool):
    global model, nlp, out_of_the_box_components
    model = model_
    nlp = spacy.load(model)
    out_of_the_box_components = list(nlp.meta.get('components'))
    if clear_mem:
        import gc
        gc.collect()
    return nlp
