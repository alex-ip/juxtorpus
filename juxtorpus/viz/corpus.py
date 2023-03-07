""" Collection of Visualisation functions for Corpus

"""
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

from juxtorpus.corpus import Corpus


def wordcloud(corpus: Corpus):
    wc = WordCloud(background_color='white', max_words=500, height=600, width=1200)
    counter = Counter(corpus.generate_words())
    for sw in stopwords.words('english'):
        try:
            del counter[sw]
        except:
            continue

    h, w = 16, 16 * 1.5

    wc.generate_from_frequencies(counter)

    plt.figure(figsize=(h, w))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def timeline(corpus: Corpus, datetime_meta: str, freq: str):
    meta = corpus.meta.get_or_raise_err(datetime_meta)
    df = pd.DataFrame([False] * len(meta.series()), index=meta.series())
    df.groupby(pd.Grouper(level=0, freq=freq)).count().plot(kind='line', figsize=(12, 6), legend=None)

    freq_to_label = {'w': 'Week', 'm': 'Month', 'y': 'Year'}
    key = freq.strip()[-1]

    plt.title(f"Count by {freq_to_label.get(key)}s")
    plt.ylabel('Count')
    plt.xlabel(freq_to_label.get(key))
    plt.show()
