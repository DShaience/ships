from typing import Tuple

import numpy as np
import pandas as pd


class Quotes:
    def __init__(self, df: pd.DataFrame):
        self.quotes = df  # expects columns ['author', 'quote']
        # self.randomState = np.random.RandomState(972)
        self.randomState = np.random.RandomState()

    def get_quote(self) -> Tuple[str, str]:
        i = self.randomState.randint(0, len(self.quotes), 1)
        author = self.quotes.loc[i, 'author'].values[0]
        quote = self.quotes.loc[i, 'quote'].values[0]
        return author, quote

    def print_quote(self):
        author, quote = self.get_quote()
        print(f"\n\"{quote}\", -{author}\n")