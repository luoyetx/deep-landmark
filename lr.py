# coding: utf-8

import sys
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    gamma, p = float(sys.argv[1]), float(sys.argv[2])

    inv = lambda k: math.pow(1+gamma*k, -p)

    x = [_ for _ in xrange(50000)]
    y = [inv(_) for _ in x]
    plt.plot(x, y)
    plt.xlabel('iter')
    plt.ylabel('scale')
    title = 'gamma={0}, p={1}'.format(gamma, p)
    plt.title(title)

    # save
    f = 'log/gamma_{0}_p_{1}.jpg'.format(gamma, p)
    plt.savefig(f)
    plt.show()
