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

## Reference

[Retrieving the images](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)

## Benchmarks:  

cs50.io     512MB RAM   
10m30s  Bottlenecks  
1m      Training    

codenvy.io  3072MB RAM  
5m45s   Bottlenecks  
1m      Training    

## Improve Performance:

[FMA, AVX, AVX2, SSE4.1, SSE4.2](https://github.com/lakshayg/tensorflow-build)