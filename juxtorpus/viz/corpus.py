""" Collection of Visualisation functions for Corpus

"""
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def wordcloud(corpus, max_words: int = 50, word_type: str = 'word'):
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

    h, w = 16, 16 * 1.5

    wc.generate_from_frequencies(counter)

    plt.figure(figsize=(h, w))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


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
    xaxis_title, yaxis_title = "Count", f"{freq_to_label.get(key, key)}"
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
    xaxis_title, yaxis_title = "Count", f"{freq_to_label.get(key, key)}"
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig
