from wordcloud import WordCloud as WC, get_single_color_func
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Callable, Set, Dict, Type

from juxtorpus.viz import Viz

# Type Aliases
PIL_HSV: Type[str] = str  # hsv(hue,sat%,value)


class PolarityWordCloud(Viz):
    """ PolarityWordCloud

    This class visualises two lists of word and score, and puts them in a wordcloud.
    The size of the words are determined by the sum of their scores.
    The colour is determined by their relative score.

    Usage:
    ```
    # quick method
    A = [('hello', 1.0), ('orange', 2.0)]
    B = [('hello', 1.0), ('apple', 0.5)]

    wc_ = PolarityWordCloud.from_(A, B, top=10)
    wc_.gradate().render()
    ```
    # custom
    df_A = pd.DataFrame(A, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    df_B = pd.DataFrame(B, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    wc_ = PolarityWordCloud(WC(background_color='white'), df_A, df_B)
    wc_.top(10)
    wc_.gradate(scheme='default').render()

    # custom colouring
    wc_ = PolarityWordCloud.from_(A, B, top=10)
    wc_.set_colours_for({'#f1959b': ('apple', 'orange')})
    wc_.colour().render()
    ```
    """

    COL_WORD: str = 'word'
    COL_SCORE: str = 'score'

    # internals
    _COL_SUMMED: str = 'summed_'
    _COL_RELATIVE: str = 'relative_'
    _COL_NORMAL: str = 'relative_normalised_'
    _COL_RELATIVE_ABS: str = 'relative_abs_'
    _COL_REL_BY_SUMMED: str = 'relative_div_summed_'
    _COL_REL_MUL_SUMMED: str = 'relative_mul_summed_'

    @staticmethod
    def from_(word_scores_A: List[Tuple[str, float]], word_scores_B: List[Tuple[str, float]]) -> 'PolarityWordCloud':
        """
        :param word_scores_A: List of words and their scores in corpus A. (Do not use negative scores!)
        :param word_scores_B: List of words and their scores in corpus B. (Do not use negative scores!)
        :param top: top
        :return: PolarityWordCloud
        """
        wc = WC(background_color='white')
        df_A = pd.DataFrame(word_scores_A, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
        df_B = pd.DataFrame(word_scores_B, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
        return PolarityWordCloud(wc, df_A, df_B)

    def __init__(self, wordcloud: WC, word_scores_df_A: pd.DataFrame, word_scores_df_B: pd.DataFrame):
        """
        Initialise a PolarityWordCloud object to compare and visualise 2 sets of words and their scores.

        :param wordcloud: wordcloud (see https://amueller.github.io/word_cloud/)
        :param word_scores_df_A: DataFrame with 'word' and 'score' columns. (Do not use negative scores!)
        :param word_scores_df_B: DataFrame with 'word' and 'score' columns. (Do not use negative scores!)
        """
        self.wc = wordcloud

        df_A = word_scores_df_A.rename(columns={PolarityWordCloud.COL_SCORE: 'score_A'})
        df_B = word_scores_df_B.rename(columns={PolarityWordCloud.COL_SCORE: 'score_B'})
        df_A[PolarityWordCloud.COL_WORD] = df_A[PolarityWordCloud.COL_WORD].str.lower()
        df_B[PolarityWordCloud.COL_WORD] = df_B[PolarityWordCloud.COL_WORD].str.lower()
        df_A = df_A.set_index(PolarityWordCloud.COL_WORD)
        df_B = df_B.set_index(PolarityWordCloud.COL_WORD)

        df = df_A.join(df_B, on=None, how='outer')
        df.fillna(value={'score_A': 0, 'score_B': 0}, inplace=True)

        self._df = df  # full df
        self._df_top_tmp = df  # df to generate wordcloud from

        # Scoring
        # main scores - relative, summed, abs(relative), abs(relative)/summed, abs(relative)*summed
        self._add_wordcloud_scores()
        # normalisation scores to between 1.0 and 2.0 - this is a hard dependency - do not switch this off
        self._add_normalised_relative_scores()

        # performance caches
        self.__top = len(self._df)
        self.__top_prev = -1

        # colouring
        self._default_colour_func = get_single_color_func("#000000")  # rgb black
        self._colour_func = self._default_colour_func
        self._colour_word_map = None
        self._colour_map_colour_funcs: List[Tuple[Callable, Set[str]]] = None

    @property
    def wordcloud(self) -> WC:
        return self.wc

    def top(self, n: int):
        """ Sets the number of words to appear on the wordcloud. Capped at maximum number of unique words. """
        if n < 0:
            raise ValueError("Must be a positive integer.")
        self.__top_prev = self.__top
        if n == self.__top:
            return self
        self.__top = n
        self._df_top_tmp = self._df.iloc[:min(n, len(self._df))]
        return self

    def render(self, height: int = 16, width: int = 16 * 1.5, title: str = ''):
        """ Renders the wordcloud on the screen. """
        fig, ax = plt.subplots(figsize=(height, width))
        ax.imshow(self.wc, interpolation='bilinear')
        ax.axis('off')
        plt.show()

    # COLOUR METHOD: MANUAL RGB MAPS

    def map_colours(self, colour_word_map: Dict[str, Set[str]]):
        """ Sets colour for particular words.
        :param colour_word_map: a dictionary mapping of rgb colour to a set of words.

        rgb colour in hex OR any PIL library supported colour string.
        """
        self._colour_map_colour_funcs = list()
        for rgb, words, in colour_word_map.items():
            self._colour_map_colour_funcs.append((get_single_color_func(rgb), words))
        self._colour_func = self._colour_map_colour_func
        return self._build()

    def _colour_map_colour_func(self, word: str, **kwargs):
        colour_func = self._find_colour_map_colour_func(word)
        return colour_func(word, **kwargs)

    def _find_colour_map_colour_func(self, word) -> Callable:
        if self._colour_map_colour_funcs is None:
            raise Exception(f"Did you call {self.map_colours.__name__}()?")
        for colour_func, words in self._colour_map_colour_funcs:
            if word in words:
                return colour_func
        return self._default_colour_func

    # COLOUR METHOD: GRADATE - put the words in a colour gradient.
    @staticmethod
    def supported_gradate_colours():
        return set(hsv_color_degrees.keys())

    def gradate(self, polar_A_colour: str, polar_B_colour: str):
        """ Puts the word cloud in gradient in accordance to the relative score. """
        _colour_word_map: Dict[str, Callable] = dict()
        for i in range(len(self._df_top_tmp)):
            word = self._df_top_tmp.iloc[i].name
            norm_score = self._df_top_tmp.iloc[i][PolarityWordCloud._COL_NORMAL]

            colour = polar_A_colour if norm_score < 1.5 else polar_B_colour
            deg = hsv_color_degrees.get(colour.upper(), None)
            if deg is None:
                raise NotImplementedError(f"Unsupported colour: {colour}. "
                                          f"Try calling {self.supported_gradate_colours.__name__}()")

            colour_func = self._norm_score_to_hsv_wrapper(score=norm_score,
                                                          degree=deg,
                                                          value=100,
                                                          min_saturation=20,
                                                          max_saturation=100)
            _colour_word_map[word] = colour_func
        self._colour_word_map = _colour_word_map
        self._colour_func = self._word_to_colour_func_wrapper
        return self._build()

    def _word_to_colour_func_wrapper(self, word, *args, **kwargs):
        colour_func = self._colour_word_map.get(word, None)
        if colour_func is None:
            raise RuntimeError("{word} is not associated with any colour func. Did you call gradate()?")
        return colour_func(word, *args, **kwargs)

    def _build(self):
        if self._state_updated():
            # expensive operation
            col_score = PolarityWordCloud._COL_REL_MUL_SUMMED
            self.wc.generate_from_frequencies(
                {self._df_top_tmp.iloc[i].name: self._df_top_tmp[col_score].iloc[i] for i in
                 range(len(self._df_top_tmp))}
            )
        self.wc.recolor(color_func=self._colour_func, random_state=42)
        return self

    def _state_updated(self):
        """ If any state is updated. Currently only if number of top words is modified. """
        # TODO: add manually re-adjusted word list. e.g. select, remove
        return self.__top_prev != self.__top

    """ Scoring
    [main]
    Relative scores - this reflects the 'polarity' of each word in the corpus. One positive, the other negative.
    Summed scores - words that are high in score for both corpus.
    [derived]
    abs(Relative) / Summed - highly polarised rare words.
    abs(Relative) * Summed - highly polarised frequent words.
    """

    def _add_wordcloud_scores(self):
        """ Compute scores derived from 'relative' and 'summed' scores. """
        self._df[PolarityWordCloud._COL_RELATIVE] = self._df['score_A'] - self._df['score_B']
        self._df[PolarityWordCloud._COL_SUMMED] = self._df['score_A'] + self._df['score_B']
        self._df[PolarityWordCloud._COL_REL_BY_SUMMED] = self._df[PolarityWordCloud._COL_RELATIVE].abs() / \
                                                         self._df[PolarityWordCloud._COL_SUMMED]
        self._df[PolarityWordCloud._COL_REL_MUL_SUMMED] = self._df[PolarityWordCloud._COL_RELATIVE].abs() * \
                                                          self._df[PolarityWordCloud._COL_SUMMED]
        self._df = self._df.sort_values(by=PolarityWordCloud._COL_SUMMED, ascending=False)

    """ Relative Score Normalisation - (for wordcloud integration)
    This is required by wordcloud as the input range must be > 0. Hence, here 1.0 to 2.0 is selected.
    Below is an important section as many output factors such as the word colour is dependent on this norm score.
    Word of warning: 
    The functions below have specific preconditions that expects the normed score. 
    So if you decide to change to extend this or make any changes to how the norm score it computed,
    make sure it is reflected in all the functions below too. Although this is intended to be
    closed/non-extendable.
    
    For detailed explanation of how this is calculated, see doc for _add_normalised_score
    """

    def _add_normalised_relative_scores(self):
        """ Normalises the values of the relative scores to be between 1 and 2 from -X to 0 to +X.
        where X can be any value dependent the scores the class is initiated with.

        Taking values with a midpoint of 0 (relative scores), it first normalises the scores between 0 and 1.
        Then move it to between 1 and 2 by summing with 1.

        The maximum absolute value of the column is taken as the boundaries. This ensures original scores of 0
        are maintained as the midpoint (now 1.5) of the normalised scores i.e. 0 -> 1.5.

        Please note: since this converts from negative scale, the 'magnitude' in the abstract sense actually increases
        from 1.5 -> 1.0. Then 1.5 -> 2.0. So 1.0 is the maximum polarity for corpus A and 2.0 for corpus B. 1.5 is mid.
        """
        _max = max(abs(self._df[PolarityWordCloud._COL_RELATIVE].min()),
                   abs(self._df[PolarityWordCloud._COL_RELATIVE].max()))
        self._df[PolarityWordCloud._COL_NORMAL] = ((self._df[PolarityWordCloud._COL_RELATIVE] - -_max) / (2 * _max)) + 1

    # Used for gradate()
    @staticmethod
    def __invert_normalised_score_to_percentage(score: float):
        """ Converts score in range 1.0 and 2.0 to 0 and 1.0.
         Where 1.5 -> 1 is increasing, so 1.5 -> 0 and 1 -> 1.
         Where 1.501 -> 2 is increasing, so 1.501 -> 0 and 2 -> 1.
        (notice it is not 1 -> 1.5 increasing as you would think normally.)

        This is used in conjunction with _add_normalised_score as it takes into account the moved axis (+ 1).
        """

        # not normalised for readability
        if 1 <= score < 1.5:
            return abs(score - 1.5) / 0.5
        elif score <= 2.0:
            return (score - 1.5) / 0.5
        else:
            raise ValueError("Score must be between 1.0 and 2.0.")

    @staticmethod
    def _norm_score_to_hsv_wrapper(score: float, degree, value, min_saturation, max_saturation) -> Callable:
        """ Return to_hsv with a different signature required by wordcloud. """

        def colour_func_wrapper(word=None, font_size=None, position=None, orientation=None, font_path=None,
                                random_state=None) -> PIL_HSV:
            return PolarityWordCloud._norm_score_to_hsv(score, degree, value, min_saturation, max_saturation)

        return colour_func_wrapper

    @staticmethod
    def _norm_score_to_hsv(score: float, deg, value, min_saturation, max_saturation) -> PIL_HSV:
        if not 0 <= min_saturation < max_saturation <= 100: raise ValueError(
            "Saturation must satisfy 0 < min < max < 100.")
        if not 1.0 <= score <= 2.0: raise ValueError("Score must satisfy 1.0 <= score <= 2.0.")

        percentage = PolarityWordCloud.__invert_normalised_score_to_percentage(score)  # move score between 0 and 1.
        sat = percentage * (max_saturation - min_saturation)  # Then use that as a percentage of the sat range.
        sat = sat + min_saturation  # move saturation axis above min_sat
        return "hsv({:.0f},{:.0f}%,{:.0f}%)".format(deg, sat, value)


hsv_color_degrees = {
    'RED': 0,
    'YELLOW': 60,
    'GREEN': 120,
    'CYAN': 180,
    'BLUE': 240,
    'MAGENTA': 300
}

if __name__ == '__main__':
    A = [('A', 0.5), ('AA', 1.0), ('AAA', 2.0), ]
    B = [('B', 0.25), ('BB', 0.5), ('BBB', 1)]

    wc_ = PolarityWordCloud.from_(A, B)
    wc_.top(10).gradate(polar_A_colour='red', polar_B_colour='blue').render()
