#!/usr/bin/env bash
### train cnns in parallel

python2.7 prototxt/generate.py CPU

# level-1
python2.7 dataset/level1.py
rm -rf log/train1.log
echo "Train LEVEL-1"
nohup python2.7 train/level.py 1 pool_on >log/train1.log 2>&1 &
# level-2
python2.7 dataset/level2.py
rm -rf log/train2.log
echo "Train LEVEL-2"
nohup python2.7 train/level.py 2 pool_on >log/train2.log 2>&1 &
# level-3
python2.7 dataset/level3.py
rm -rf log/train3.log
echo "Train LEVEL-3"
nohup python2.7 train/level.py 3 pool_on >log/train3.log 2>&1 &

echo "=.="
