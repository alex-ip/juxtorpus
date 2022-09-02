""" Spacy Pipelines

A collection of spacy pipeline components for different use cases.
This will improve the efficiency of the library. Efficiency are compute time and memory usage.

Reduce Memory usage:
Spacy allows you to 'include' and exclude components. The tradeoff is that the spacy nlp model needs to be reloaded.

Improve computation/inference time:
Spacy allows you to 'disable'/'enable' components in your models. This skips parts of the pipeline hence reduces
computation time. But it does not save memory.

https://spacy.io/usage/processing-pipelines


Things to note:
The quick and easy way to use a pretrained pipeline from spacy is via spacy.load('...'), this creates a Language object
configured with their config.cfg and load a bunch of models from binaries.
(You may go to spacy.utils - load_model_from_init_py, load_model_from_path if you want to dig deeper)

Design:
Assuming we use an out-of-the-box pipeline from spaCy, the models are going to be loaded to RAM from disk.
What this means is that we can't just call 'nlp.remove_pipe' then 'nlp.add_pipe' on an out-of-the-box component.
If we do, we need to load the model back on ourselves with the config from the package,
i.e. call from_disk that component (note: some components inherit from_disk directly from TrainablePipe class).
This is possible to do/hack if we wrap some logic around spacy.load(...exclude=[]), just exclude everything else and
repopulate the pipeline. But for our case, I think I'll just keep it simple for now.
What this means is in terms of the design:
Out-of-the-box components are 'disabled' ONLY when not in use.
Custom components are 'added' and 'removed' from our pipelines as we see fit. These are appended.

Future:
1. This simple behaviour is a tech debt. If we have our own models later and write our own config.cfg + load models from
disk, it'll be a problem; Both with the current behaviour and the RAM usage.
Trainable components need to override from_disk() method to load the models.

2. Ordering of components are not incorporated. (todo: add an 'after' and 'before' attribute to the Components? - see nlp.analyze_pipes)


Other resources:
https://spacy.io/models#design

Notes:
    + use 'senter' instead of 'parser' when dependency parsing is not required. Inference is ~10x faster.
"""

# Default included pipelines
spacy_default = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

# Inference pipelines ONLY. Place this array in the 'disable' parameter in nlp.pipe(,disable=)
disable_basic = spacy_default  # used if you only require frequencies from the docs.

from typing import List
from spacy import Language
from juxtorpus.components.hashtags import HashtagComponent


@Language.factory("extract_hashtags")
def create_hashtag_component(nlp: Language, name: str):
    return HashtagComponent(nlp, name)


# todo: ordering of components not handled here
# todo: trainable components + modelling not entirely thought through here.


def adjust_pipeline_with(nlp: Language, components: List[str]):
    """ add pipes if not exist. """
    from juxtorpus import out_of_the_box_components  # hacky
    for name in out_of_the_box_components:
        nlp.disable_pipe(name)

    for c in components:
        if c not in out_of_the_box_components and c in nlp.component_names:
            nlp.remove_pipe(c)

    for c in components:
        if c in out_of_the_box_components:
            nlp.enable_pipe(c)
        else:
            _ = nlp.add_pipe(factory_name=c)


if __name__ == '__main__':
    from juxtorpus import nlp

    print(nlp.pipe_names)
    print(nlp("hello"))
    new_pipe_names = ['tok2vec', 'ner', 'extract_hashtags']
    adjust_pipeline_with(nlp, new_pipe_names)
    print(nlp.pipe_names)
    print(nlp("hello #atap")._.hashtags)
    adjust_pipeline_with(nlp, ['ner'])
    print(nlp("hello"))
