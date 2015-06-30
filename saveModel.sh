#!/usr/bin/env bash

echo $1

root=../landmarkModel/$1
mkdir $root
cp -r log $root
cp -r model $root
cp -r prototxt $root

