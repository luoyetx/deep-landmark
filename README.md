landmark
========

Predict facial landmarks with Deep CNNs powered by Caffe.

### Data

All training data can be downloaded from [here](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm).

Download the images and extract to `dataset` with `train` and `test`.


### Train

```
./bootstrap.sh
```

This will first generate prototxt files for caffe models and convert training data(images and landmarks) into h5 files. Then We will train the level-1 CNNs and use the result to generate training data for level-2. And for level-2 and level-3 goes the same way.

### Models

All model files are under `model`, we can modify `*.template` file to change the caffe model for every level.

### References

1. [Caffe](http://caffe.berkeleyvision.org/)
2. [Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
