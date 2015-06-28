#!/usr/bin/env bash

python2.7 prototxt/generate.py

# level-1
python2.7 dataset/level1.py
rm -rf log/train1.log
echo "Train LEVEL-1"
python2.7 train/level.py 1
# level-2
python2.7 dataset/level2.py
rm -rf log/train2.log
echo "Train LEVEL-2"
python2.7 train/level.py 2
# level-3
python2.7 dataset/level3.py
rm -rf log/train3.log
echo "Train LEVEL-3"
python2.7 train/level.py 3

echo "=.="
