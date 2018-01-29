# floyd run --data efcic/datasets/inception/1:inception --data efcic/datasets/2flowers/1:flowers "bash run.sh"

mkdir /tmp/imagenet
cp /inception/* /tmp/imagenet
cd /code

python retrain.py \
  --bottleneck_dir=/output/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=/tmp/imagenet \
  --output_graph=/output/retrained_graph.pb \
  --output_labels=/output/retrained_labels.txt \
  --image_dir=/flowers/flower_photos