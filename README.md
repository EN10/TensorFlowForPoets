# TensorFlow For Poets

## Install Tensorflow:

    sudo pip install -U pip  
    sudo pip install tensorflow 

## Download Flowers:

    curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
    tar xzf flower_photos.tgz

## Speedup Training 
reduce the number of images by 70%    

    ls flower_photos/roses | wc -l
    rm flower_photos/*/[3-9]*
also only use 2 flowers e.g. roses and sunflowers

## Training

    tensorboard --logdir training_summaries --port 8080 &
    python retrain.py   --bottleneck_dir=bottlenecks   --how_many_training_steps=500   --model_dir=inception  --summaries_dir=training_summaries/basic   --output_graph=retrained_graph.pb   --output_labels=retrained_labels.txt   --image_dir=flower_photos

## Classifying an image

    python label_image.py flower_photos/roses/2414954629_3708a1a04d.jpg 

## Performance
Precompiled with FMA, AVX, AVX2, SSE4.1, SSE4.2  
Working on codenvy.io not on c9.io  

    sudo pip install --ignore-installed --upgrade tensorflow-1.2.1-cp27-none-linux_x86_64.whl
    
Precompiled bottlenecks also included as tgz.

## Reference

* [Retrieving the images](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)  
* [FMA, AVX, AVX2, SSE4.1, SSE4.2](https://github.com/lakshayg/tensorflow-build)
* [Community wheels](https://github.com/yaroslavvb/tensorflow-community-wheels)

## Benchmarks:  
pip Tensorflow:
cs50.io     512MB RAM   
10m30s  Bottlenecks  
1m      Training    

codenvy.io  3072MB RAM  
5m45s   Bottlenecks  
1m      Training    

Build supporting AVX, AVX2, FMA, SSE4.1, SSE4.2:  
codenvy.io  2048MB RAM  same as 3072MB RAM  
2m20s   Bottlenecks  
1m      Training 

## Build from Source
* [Docker Ubuntu](https://hub.docker.com/_/ubuntu/)
* [Install Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu)
* [Build from Source](https://www.tensorflow.org/install/install_sources#clone_the_tensorflow_repository)
* [Add SSE & AVX](https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions)

Build:

    bazel build -c opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma //tensorflow/tools/pip_package:build_pip_package