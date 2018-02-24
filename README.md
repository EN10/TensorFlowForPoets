# TensorFlow For Poets

Retraining one of Google's CNN image classification models to new categories using Transfer Learning.  
This can be an much faster (in a few minutes) than training from scratch (Inception V3 took Google, 2 weeks).

Based on [Tensorflow Retrain](https://www.tensorflow.org/versions/master/tutorials/image_retraining) and [Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0) (updated Dec 6 2017)

### Contents

 [Install Tensorflow](#install-tensorflow)  
 [Download Flowers](#download-flowers)  
 [Speedup Training](#speedup-training)  
 [(Re)Training](#retraining)    
 [Classifying an Image](#classifying-an-image)  
 [Training on Your Own Categories](#training-on-your-own-categories)    
 [Benchmarks](#benchmarks)  
 [Compiled](#compiled)  
 [Build from Source](#build-from-source)    

## Install Tensorflow:

    sudo pip install -U pip  
    sudo pip install tensorflow 

## Download Flowers:
    
    mkdir tf_files
    cd tf_files
    curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
    tar xzf flower_photos.tgz

* [Retrieving the images](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#2)  

## Speedup Training 
reduce the number of images by ~70%    

    ls flower_photos/roses | wc -l
    rm flower_photos/*/[3-9]*
also only use 2 flowers e.g. roses and sunflowers  

    rm flower_photos/daisy/ flower_photos/dandelion/ flower_photos/tulips/ -r

* Precompiled bottlenecks also included as tgz.
* **[Floydhub](https://github.com/EN10/FloydHub)** [run.sh](https://github.com/EN10/TensorFlowForPoets/blob/master/run.sh) *Intel Xeon® · 2 Cores   8 GB RAM*
* **colab** *K80 12GB* [colab.sh](https://github.com/EN10/TensorFlowForPoets/blob/master/colab.sh)

colab code:

    !wget https://raw.githubusercontent.com/EN10/TensorFlowForPoets/master/colab.sh
    !bash colab.sh

## (Re)Training

Download retrain file:
    
    wget https://raw.githubusercontent.com/EN10/TensorFlow-For-Poets/master/retrain.py

**MobileNet 0.5**:  Faster (< 2m) Less Accurate (Top-1 64%)

    python retrain.py \
      --bottleneck_dir=tf_files/bottlenecks \
      --how_many_training_steps=500 \
      --model_dir=tf_files/models/ \
      --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 \
      --output_graph=tf_files/retrained_graph.pb \
      --output_labels=tf_files/retrained_labels.txt \
      --architecture=mobilenet_0.50_224 \
      --image_dir=tf_files/flower_photos

`mobilenet` can be configured from `0.50` to `1.0`, more accuracy but slower.

**Inception V3**:   Slower (5 - 20m) More Accurate (Top-1 78%)

    python retrain.py \
      --bottleneck_dir=tf_files/bottlenecks \
      --how_many_training_steps=500 \
      --model_dir=tf_files/models/inception_v3 \
      --output_graph=tf_files/retrained_graph.pb \
      --output_labels=tf_files/retrained_labels.txt \
      --image_dir=tf_files/flower_photos

* [(Re)training Inception](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)  

## Classifying an Image

Download label_image file:

    wget https://raw.githubusercontent.com/EN10/TensorFlow-For-Poets/master/label_image.py

Predict image label:

    python label_image.py \
    --graph=tf_files/retrained_graph.pb  \
    --image=image.jpg

* [Label Image](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4)  

## Training on Your Own Categories

`retrain.py` uses `--image_dir` as the root folder for training.  
Each sub-folder is named after one of your categories and contains only images from that category.  
* [Training on Your Own Categories](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#7)  

For training to work well, you should gather at least a hundred photos of each kind of object you want to recognize.  
* [Creating a Set of Training Images](https://www.tensorflow.org/tutorials/image_retraining#creating_a_set_of_training_images)  

Tool to download images for training:
* [Image Batch Downloader](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en)

## Benchmarks:  
`rm [3-9]*` & 2 Flowers: roses and sunflowers (aka slim)    
282 roses & 304 sunfowers i.e. 586 bottlenecks

| Model | PAAS | RAM | OS | Tensorflow | CPU | Performance | Notes |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| MobileNet 0.5 | cs50.io  | 512MB | 14.04.5 | 1.4.1 | Not Compiled | 1m20s | |
| MobileNet 0.5 | cs50.io  | 1024MB | 14.04.5 | 1.4.1 | Not Compiled | 1m20s | |
| MobileNet 0.5 | colab  | 1024MB | 17.10 | 1.6.0rc1 | Compiled | 43s | |
| MobileNet 0.5 | colab  | 12GB | 17.10 | 1.6.0rc1 | K80 | 24s | |
| Inception v3 | cs50.io  | 512MB | 14.04.5 | 1.4.1 | Not Compiled | 15m | |
| Inception v3 | cs50.io  | 512MB | 14.04.5 | 1.2.1 | Compiled | 6m30s | |
| Inception v3 | floydhub  | 8GB | 16.04.2 | 1.1.0 | Compiled | 4m30s | with install |
| Inception v3 | floydhub  | 8GB | 16.04.2 | 1.1.0 | Compiled | 4m | with datasets |
| Inception v3 | codenvy.io  | 3072MB | 16.04 | 1.2.1 | Not Compiled | 6m45s | |
| Inception v3 | codenvy.io  | 3072MB | 16.04 | 1.2.1 | Compiled | 3m20s | |
| Inception v3 | codenvy.io  | 2048MB | 16.04 | 1.2.1 | Compiled | 3m20s | |
| Inception v3 | colab | 1024MB | 17.10 | 1.6.0rc1 | Compiled | 3m38s | |
| Inception v3 | colab | 12GB | 17.10 | 1.6.0rc1 | K80 | 1m23s | |

Performance = Bottlenecks + Training where Training ~ 1 Min     
CPU: `TensorFlow binary compiled to use: SSE4.1 SSE4.2 AVX`     
Precompiled Bottlenecks in `/inception_bottlenecks` (slim see above)

* [Workspace Benchmark](https://github.com/EN10/WorkspaceCPUBench#benchmarks)

## Compiled
Precompiled with FMA, AVX, AVX2, SSE4.1, SSE4.2:  
* [FloydHub EN10](https://github.com/EN10/FloydHub)

* [FMA, AVX, AVX2, SSE4.1, SSE4.2 Builds](https://github.com/lakshayg/tensorflow-build) Working on codenvy.io not on c9.io  
* [Precompiled Example for 14.04](https://github.com/EN10/KerasCIFAR#performance)

`wget https://github.com/EN10/BuildTF/raw/771df48529285c69ef760327121e996750b3916e/tensorflow-1.4.0-cp27-none-linux_x86_64.whl`   
`sudo pip install --ignore-installed --upgrade tensorflow-1.4.0-cp27-none-linux_x86_64.whl`  
    
OS: Ubuntu 14.04.5 LTS - GCC version 4.8.4 - Python: 2.7.6 - Tensorflow 1.2.1

## Build from Source
* [Docker Ubuntu](https://hub.docker.com/_/ubuntu/)
* [Install Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu)
* [Build from Source](https://www.tensorflow.org/install/install_sources#clone_the_tensorflow_repository)
* [Add SSE & AVX](https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions)

Build:

    bazel build --config=opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma //tensorflow/tools/pip_package:build_pip_package