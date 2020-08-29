from typing import Tuple
import numpy as np
import pandas as pd

#
# class Quotes:
#     def __init__(self, df: pd.DataFrame):
#         self.quotes = df  # expects columns ['author', 'quote']
#         self.randomState = np.random.RandomState()
#
#     def get_quote(self) -> Tuple[str, str]:
#         i = self.randomState.randint(0, len(self.quotes), 1)
#         author = self.quotes.loc[i, 'author'].values[0]
#         quote = self.quotes.loc[i, 'quote'].values[0]
#         return author, quote
#
#     def print_quote(self):
#         author, quote = self.get_quote()
#         print(f"\n\"{quote}\", -{author}\n")
#


class QuotesSingleton(object):
    """
    Quotes class as singleton ensures that I can print silly quotes throughout the code
    """
    __instance = None

    def __new__(cls, quotes_path: str = None):
        if cls.__instance is None:
            cls.__instance = super(QuotesSingleton,cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, quotes_path: str = None):
        if self.__initialized:
            return
        self.__initialized = True

        self.quotes = pd.read_csv(quotes_path)
        self.randomState = np.random.RandomState()

    def get_quote(self) -> Tuple[str, str]:
        i = self.randomState.randint(0, len(self.quotes), 1)
        author = self.quotes.loc[i, 'author'].values[0]
        quote = self.quotes.loc[i, 'quote'].values[0]
        return author, quote

    def print_quote(self):
        author, quote = self.get_quote()
        msg = f"\n\"{quote}\", -{author}\n"
        wrapper = '='*len(msg)
        print(f"\n{wrapper}\n\"{quote}\", -{author}\n{wrapper}\n")
