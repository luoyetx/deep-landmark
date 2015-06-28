#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file generate prototxt file for LEVEL-2 and LEVEL-3
"""

import sys


nameDict = {
    's0': ['F'],
    's1': ['EN', 'NM'],
    's3': ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'],}

def generate(network, level, names, mode='GPU'):
    """
        Generate template
        network: CNN type
        level: LEVEL
        names: CNN names
        mode: CPU or GPU
    """
    assert(mode == 'GPU' or mode == 'CPU')

    types = ['train', 'solver', 'deploy']
    for name in names:
        for t in types:
            tempalteFile = 'prototxt/{0}_{1}.prototxt.template'.format(network, t)
            with open(tempalteFile, 'r') as fd:
                template = fd.read()
                outputFile = 'prototxt/{0}_{1}_{2}.prototxt'.format(level, name, t)
                with open(outputFile, 'w') as fd:
                    fd.write(template.format(level=level, name=name, mode=mode))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mode = 'GPU'
    else:
        mode = 'CPU'
    generate('s0', 1, nameDict['s0'], mode)
    generate('s1', 1, nameDict['s1'], mode)
    generate('s3', 2, nameDict['s3'], mode)
    generate('s3', 3, nameDict['s3'], mode)
    # Done
