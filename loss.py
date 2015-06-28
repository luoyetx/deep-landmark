# coding: utf-8

import os, sys
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    f = sys.argv[1]

    title = os.path.splitext(f)[0]
    df = pd.read_csv(f)
    # filter
    df = df[df['iteration'] > 5000]
    df.plot(x='iteration', y='loss', title=title)
    plt.savefig('%s.png'%title)
    plt.show()
