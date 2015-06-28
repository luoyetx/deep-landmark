#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file train Caffe CNN models
"""

import os, sys
import multiprocessing


pool_on = False

models = [
    ['F', 'EN', 'NM'],
    ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'],
    ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'],]

def w(c):
    if c != 0:
        print '\n'
        print ':-('
        print '\n'
        sys.exit()

def runCommand(cmd):
    w(os.system(cmd))

def train(level=1):
    """
        train caffe model
    """
    cmds = []
    for t in models[level-1]:
        cmd = 'mkdir model/{0}_{1}'.format(level, t)
        os.system(cmd)
        cmd = 'caffe train --solver prototxt/{0}_{1}_solver.prototxt'.format(level, t)
        # w(os.system(cmd))
        cmds.append('caffe train --solver prototxt/{0}_{1}_solver.prototxt'.format(level, t))
    # we train level-2 and level-3 with mutilprocess (we may train two level in parallel)
    if level > 1 and pool_on:
        pool_size = 3
        pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
        pool.map(runCommand, cmds)
        pool.close()
        pool.join()
    else:
        for cmd in cmds:
            runCommand(cmd)


if __name__ == '__main__':
    argc = len(sys.argv)
    assert(2 <= argc <= 3)
    if argc == 3:
        pool_on = True

    level = int(sys.argv[1])
    if 1 <= level <= 3:
        train(level)
    else:
        for level in range(1, 4):
            train(level)
