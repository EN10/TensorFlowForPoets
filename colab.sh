mkdir tf_files
cd tf_files
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

# - 70% images and 2 flowers speed up training
rm flower_photos/*/[3-9]*
rm flower_photos/daisy/ flower_photos/dandelion/ flower_photos/tulips/ -r

wget https://raw.githubusercontent.com/EN10/TensorFlow-For-Poets/master/retrain.py

python retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture=mobilenet_0.50_224 \
  --image_dir=flower_photos