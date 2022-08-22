import wordcloud
from wordcloud import WordCloud as WC, get_single_color_func
import matplotlib.pyplot as plt
import pandas as pd

"""
Input: list of words with a score (e.g. frequency, tfidf etc)

This is a visualisation class.
"""

from typing import List, Tuple, Callable, Set, Dict

from juxtorpus.viz import Viz


def tuples_to_dict(tuples: List[Tuple[str, float]]):
    dict_ = dict()
    for k, s in tuples:
        dict_[k] = s
    return dict_


class PolarityWordCloud(Viz):
    """ PolarityWordCloud

    """

    @staticmethod
    def from_(word_scores_A: List[Tuple[str, float]], word_scores_B: List[Tuple[str, float]]) -> 'PolarityWordCloud':
        """
        :param word_scores: List of words and their scores. (Expects both positive and negative scores)
        :return: PolarityWordCloud
        """
        # todo: concat and convert one of the scores to be negative
        wc = WC(background_color='white')
        # df = pd.DataFrame(word_scores_A, columns=['word', 'score'])
        df_A = pd.DataFrame(word_scores_A, columns=['word', 'score_A']).set_index('word')
        df_B = pd.DataFrame(word_scores_B, columns=['word', 'score_B']).set_index('word')
        df = df_A.join(df_B, on=None, how='outer')
        df.fillna(value={'score_A': 0, 'score_B': 0}, inplace=True)
        df['__relative__'] = df['score_A'] - df['score_B']
        df['__summed__'] = df['score_A'] + df['score_B']
        PolarityWordCloud._add_normalise_scores(df, '__relative__')
        return PolarityWordCloud(wc, df)

    @staticmethod
    def _add_normalise_scores(df: pd.DataFrame, col_score: str):
        # for negative scores, we need to move the axis - maybe it'd be easier to use pandas.Series instead.
        _max = max(abs(df[col_score].min()), abs(df[col_score].max()))
        df['__normalised__'] = ((df[col_score] - -_max) / (2 * _max)) + 1

    def __init__(self, wordcloud: WC, word_scores_df: pd.DataFrame):
        # TODO: refactor this to accept 2 dataframes instead of relying on positive and negative scores?
        self.wc = wordcloud
        self.df = word_scores_df
        self._colour_funcs: List[Tuple[Callable, Set[str]]] = None
        self._default_colour_func = get_single_color_func(HEX_OFFWHITE)

    @property
    def wordcloud(self) -> WC:
        return self.wc

    def set_colours_with(self, condition_map: Dict[str, Callable[[Tuple[str, float]], bool]], col_score: str):
        """ Sets colour for words satisfying the specified condition."""

        # build the internal colour_map
        _colour_word_map: Dict[str, Set[str]] = dict()
        for i in range(len(self.df)):
            for colour, condition in condition_map.items():
                if condition(self.df.iloc[i][col_score]):
                    word_set = _colour_word_map.get(colour, set())
                    word_set.add(self.df.iloc[i].name)
                    _colour_word_map[colour] = word_set
                    break

        self.set_colours_for(_colour_word_map)

    def set_colours_for(self, colour_word_map: Dict[str, Set[str]]):
        """ Sets colour for particular words """
        self._colour_funcs = list()
        for colour, words, in colour_word_map.items():
            self._colour_funcs.append((get_single_color_func(colour), words))

    def gradate(self, scheme: str = ''):
        """ Puts the word cloud in gradient in accordance to the score.

        For more complex colouring, use set_colours_with or set_colours_for.
        This uses set_colours_with internally.
        """

        self.wc.generate_from_frequencies(
            {self.df.iloc[i].name: self.df.iloc[i]['__summed__'] for i in range(len(self.df))}
        )

        # set colour based on words
        self._set_normalised_default_colour_scheme()

        self.wc.recolor(color_func=self._gradate_colour_func)
        return self

    def render(self, height: int = 24, width: int = 24 * 1.5):
        """ Renders the wordcloud on the screen. """
        fig, ax = plt.subplots(figsize=(height, width))
        ax.imshow(self.wc, interpolation='bilinear')
        ax.axis('off')
        plt.show()

    def _gradate_colour_func(self, word: str, **kwargs):
        colour_func = self._find_colour_func(word)
        return colour_func(word, **kwargs)

    def _find_colour_func(self, word) -> Callable:
        if self._colour_funcs is None:
            raise Exception("Did you call set_colour_with or set_colour_for?")  # TODO
        for colour_func, words in self._colour_funcs:
            if word in words:
                return colour_func
        return self._default_colour_func

    def _set_normalised_default_colour_scheme(self):
        """ Sets the default colours based on score values. Used with add_normalised_scores static method."""
        self.set_colours_with({
            HEX_GREEN_0: lambda norm_score: 1.52 < norm_score <= 1.6,
            HEX_GREEN_1: lambda norm_score: 1.6 < norm_score <= 1.7,
            HEX_GREEN_2: lambda norm_score: 1.7 < norm_score <= 1.8,
            HEX_GREEN_3: lambda norm_score: 1.8 < norm_score <= 2.0,
            HEX_BROWN: lambda norm_score: 1.48 <= norm_score <= 1.52,
            HEX_RED_0: lambda norm_score: 1.4 <= norm_score < 1.48,
            HEX_RED_1: lambda norm_score: 1.3 <= norm_score < 1.4,
            HEX_RED_2: lambda norm_score: 1.2 <= norm_score < 1.3,
            HEX_RED_3: lambda norm_score: 1 <= norm_score < 1.2,
        }, col_score='__normalised__')


HEX_BLACK = "#000000"
HEX_OFFWHITE = "#F8F0E3"
HEX_GREEN_0 = "#abe098"
HEX_GREEN_1 = "#83d475"
HEX_GREEN_2 = "#57c84d"
HEX_GREEN_3 = "#2eb62c"
HEX_RED_0 = "#f1959b"
HEX_RED_1 = "#f07470"
HEX_RED_2 = "#ea4c46"
HEX_RED_3 = "#dc1c13"
HEX_BROWN = "#722F37"

if __name__ == '__main__':
    A = [
        ('hello', 1.0),
        ('there', 1.0),
        ('orange', 2.0),
    ]

    B = [
        ('hello', 1.0),
        ('bye', 0.25),
        ('apple', 0.5)
    ]

    wc_ = PolarityWordCloud.from_(A, B)  # green, red
    print(wc_.df)
    wc_.gradate().render()
