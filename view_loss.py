# coding: utf-8

import os, sys
import pandas as pd
import matplotlib.pyplot as plt


def plot(f):
    """
        plot loss curve
    """
    title = os.path.splitext(f)[0]
    df = pd.read_csv(f)
    # filter
    df = df[df['iteration'] > 10000]
    df.plot(x='iteration', y='loss', title=title)
    plt.savefig('%s.png'%title)

if __name__ == '__main__':
    for f in os.listdir('log'):
        if os.path.splitext(f)[-1] == '.csv':
            plot(os.path.join('log', f))
