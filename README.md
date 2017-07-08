# TensorFlow For Poets

## Install Tensorflow:

    sudo pip install -U pip  
    sudo pip install tensorflow 

[Retrieving the images](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)

speedup training, reduce the number of images by 70%
rm flower_photos/*/[3-9]*
also only use 2 flowers e.g. roses and sunflowers

tensorboard --logdir training_summaries --port 8080 &

python retrain.py   --bottleneck_dir=bottlenecks   --how_many_training_steps=500   --model_dir=inception  --summaries_dir=training_summaries/basic   --output_graph=retrained_graph.pb   --output_labels=retrained_labels.txt   --image_dir=flower_photos

cs50.io     512MB RAM
10m30s  Bottlenecks
1m      Training

codenvy.io  3072MB RAM
5m45s   Bottlenecks
1m      Training