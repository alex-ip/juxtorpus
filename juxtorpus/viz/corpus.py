""" Collection of Visualisation functions for Corpus

"""
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import math


def wordclouds(corpora, names: list[str], max_words: int = 50, word_type: str = 'word'):
    MAX_COLS = 2
    nrows = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=MAX_COLS, figsize=(16, 16 * 1.5))
    r, c = 0, 0
    for name in names:
        assert corpora[name], f"{name} does not exist in Corpora."
        corpus = corpora[name]
        wc = _wordcloud(corpus, max_words, word_type)
        if nrows == 1:
            ax = axes[c]
        else:
            ax = axes[r][c]
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        if c == MAX_COLS - 1: r += 1
        c = (c + 1) % MAX_COLS

    plt.tight_layout(pad=0)
    plt.show()


def wordcloud(corpus, max_words: int = 50, word_type: str = 'word'):
    wc = _wordcloud(corpus, max_words, word_type)
    h, w = 16, 16 * 1.5
    plt.figure(figsize=(h, w))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def _wordcloud(corpus, max_words, word_type):
    modes = {'word', 'hashtag', 'mention'}
    wc = WordCloud(background_color='white', max_words=max_words, height=600, width=1200)
    if word_type == 'word':
        generator = corpus.generate_words()
    elif word_type == 'hashtag':
        generator = corpus.generate_hashtags()
    elif word_type == 'mention':
        generator = corpus.generate_mentions()
    else:
        raise ValueError(f"Mode {word_type} is not supported. Must be one of {', '.join(modes)}")
    counter = Counter(generator)
    for sw in stopwords.words('english'):
        try:
            del counter[sw]
        except:
            continue

    wc.generate_from_frequencies(counter)
    return wc


def timeline(corpus, datetime_meta: str, freq: str):
    meta = corpus.meta.get_or_raise_err(datetime_meta)
    df = pd.DataFrame([False] * len(meta.series()), index=meta.series())
    df = df.groupby(pd.Grouper(level=0, freq=freq)).count()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index.tolist(), y=df[0].tolist(), name=meta.id, showlegend=True)
    )

    freq_to_label = {'w': 'Week', 'm': 'Month', 'y': 'Year'}
    key = freq.strip()[-1]

    title = f"Count by {freq_to_label.get(key, key)}"
    xaxis_title, yaxis_title = f"{freq_to_label.get(key, key)}", "Count"
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def timelines(corpora, names: list[str], datetime_meta: str, freq: str):
    # datetime_series = None
    for name in names:
        corpus = corpora[name]
        assert corpus, f"{name} does not exist in corpora."
        # meta = corpus.meta.get_or_raise_err(datetime_meta)
        # if not datetime_series: datetime_series = meta.series()
    fig = go.Figure()
    for name in names:
        corpus = corpora[name]
        meta = corpus.meta.get_or_raise_err(datetime_meta)
        df = pd.DataFrame([False] * len(meta.series()), index=meta.series())
        df = df.groupby(pd.Grouper(level=0, freq=freq)).count()
        fig.add_trace(
            go.Scatter(x=df.index.tolist(), y=df[0].tolist(), name=name, showlegend=True)
        )
    freq_to_label = {'w': 'Week', 'm': 'Month', 'y': 'Year'}
    key = freq.strip()[-1]

    title = f"Count by {freq_to_label.get(key, key)}"
    xaxis_title, yaxis_title = f"{freq_to_label.get(key, key)}", "Count"
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig
