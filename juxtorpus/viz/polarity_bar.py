import pandas as pd
from typing import List, Tuple, Callable, Set, Dict
import matplotlib.pyplot as plt

from juxtorpus.viz import Viz


class PolarityBar(Viz):
    @staticmethod
    def from_(word_scores: List[Tuple[str, float]]) -> 'PolarityBar':
        df = pd.DataFrame(word_scores, columns=['word', 'score'])
        return PolarityBar(word_score_df=df)

    def __init__(self, word_score_df: pd.DataFrame):
        self._df = word_score_df
        self._stacked_in_subplot: bool = False
        self._relative_in_subplot: bool = False

    def stack(self):
        """ Set up the bar visuals as stacked bars """

        _positives = self._df[self._df['score'] >= 0]
        _negatives = self._df[self._df['score'] < 0]

        fig, ax = plt.subplots()
        b1 = ax.barh(_positives['word'], _positives['score'], color='green')
        b2 = ax.barh(_negatives['word'], _negatives['score'], color='red')

        plt.legend([b1, b2], ['CorpusA', 'CorpusB'], loc='upper right')
        plt.title("Stacked Frequency")
        return self

    def relative(self):
        """ Set up the bar visuals as a relative bar """
        fig, ax = plt.subplots()
        self._df['__viz__positive'] = self._df['score'] > 0
        _sorted_df = self._df.sort_values(by='score')
        b = ax.barh(_sorted_df['word'], _sorted_df['score'],
                    color=_sorted_df['__viz__positive'].map({True: 'g', False: 'r'}))
        plt.title("Relative Frequency Differences")
        return self

    def render(self):
        # show the subplots
        plt.show()
        self._cleanup()

    def _cleanup(self):
        if '__viz__positive' in self._df.columns:
            self._df.drop(columns=['__viz__positive'], inplace=True)


if __name__ == '__main__':
    word_scores = [
        ('hello', -0.25),
        ('there', 1.0),
    ]
    pbar = PolarityBar.from_(word_scores)
    pbar.relative().render()
    pbar.stack().render()
