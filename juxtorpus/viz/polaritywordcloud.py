from wordcloud import WordCloud as WC, get_single_color_func
import matplotlib.pyplot as plt
import pandas as pd

"""
Input: list of words with a score (e.g. frequency, tfidf etc)

This is a visualisation class.
"""

from typing import List, Tuple, Callable, Set, Dict


def tuples_to_dict(tuples: List[Tuple[str, float]]):
    dict_ = dict()
    for k, s in tuples:
        dict_[k] = s
    return dict_


class PolarityWordCloud(object):

    @staticmethod
    def from_(word_scores: List[Tuple[str, float]]) -> 'PolarityWordCloud':
        wc = WC(background_color='white')
        df = pd.DataFrame(word_scores, columns=['word', 'score'])
        PolarityWordCloud._add_normalise_scores(df)
        return PolarityWordCloud(wc, df)

    @staticmethod
    def _add_normalise_scores(df: pd.DataFrame):
        # for negative scores, we need to move the axis - maybe it'd be easier to use pandas.Series instead.
        df['normalised'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min()) + 1

    def __init__(self, wordcloud: WC, word_scores_df: pd.DataFrame):
        self.wc = wordcloud
        self.df = word_scores_df
        self._colour_funcs: List[Tuple[Callable, Set[str]]] = None
        self._default_colour_func = get_single_color_func(HEX_BLACK)

    @property
    def wordcloud(self) -> WC:
        return self.wc

    def set_colours_with(self, condition_map: Dict[str, Callable[[Tuple[str, float]], bool]]):
        """ Sets colour for words satisfying the specified condition."""

        # build the internal colour_map
        _colour_map: Dict[str, Set[str]] = dict()
        for i in range(len(self.df)):
            for colour, condition in condition_map.items():
                if condition(self.df.iloc[i]['normalised']):
                    word_set = _colour_map.get(colour, set())
                    word_set.add(self.df.iloc[i]['word'])
                    _colour_map[colour] = word_set
                    break
        self._colour_funcs = list()
        for colour, words in _colour_map.items():
            self._colour_funcs.append((get_single_color_func(colour), words))

    def set_colours_for(self, word_map: Dict[str, Set[str]]):
        """ Sets colour for particular words """

        # builds the colour func map
        pass

    def gradate(self, scheme: str = ''):
        """ Puts the word cloud in gradient in accordance to the score.

        For more complex colouring, use set_colours_with or set_colours_for.
        This uses set_colours_with internally.
        """

        self.wc.generate_from_frequencies(
            {self.df.iloc[i]['word']: self.df.iloc[i]['normalised'] for i in range(len(self.df))}
        )

        self.set_colours_with({
            HEX_GREEN_0: lambda score: 1.5 < score <= 1.6,
            HEX_GREEN_1: lambda score: 1.6 < score <= 1.7,
            HEX_GREEN_2: lambda score: 1.7 < score <= 1.8,
            HEX_GREEN_3: lambda score: 1.8 < score <= 2.0,
            HEX_RED_0: lambda score: 1.4 <= score <= 1.5,
            HEX_RED_1: lambda score: 1.3 <= score < 1.4,
            HEX_RED_2: lambda score: 1.2 <= score < 1.3,
            HEX_RED_3: lambda score: 1 <= score < 1.2,
        })

        self.wc.recolor(color_func=self._gradate_colour_func)
        return self

    def render(self):
        """ Renders the wordcloud on the screen. """
        plt.imshow(self.wc, interpolation='bilinear')
        plt.axis('off')
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


HEX_BLACK = "#000000"
HEX_GREEN_0 = "#abe098"
HEX_GREEN_1 = "#83d475"
HEX_GREEN_2 = "#57c84d"
HEX_GREEN_3 = "#2eb62c"
HEX_RED_0 = "#f1959b"
HEX_RED_1 = "#f07470"
HEX_RED_2 = "#ea4c46"
HEX_RED_3 = "#dc1c13"

if __name__ == '__main__':
    word_scores = [
        ('hello', -0.25),
        ('there', 1.0)
    ]

    wc_ = PolarityWordCloud.from_(word_scores)
    print(wc_.df)
    wc_.gradate().render()
