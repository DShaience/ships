import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def series_hist(s1: pd.Series, s2: pd.Series, s1_label: str, s2_label: str, xlabel: str, title: str, toShow: bool = True):
    bins = 300
    f, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=True)
    sns.distplot(s1, color="darkred", ax=axes, bins=bins, label=s1_label, hist_kws=dict(alpha=0.8))
    sns.distplot(s2, color="navy", ax=axes, bins=bins, label=s2_label, hist_kws=dict(alpha=0.8))
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(title)
    plt.legend()
    if toShow:
        plt.show()
    else:
        plt.close(f)