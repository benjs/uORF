#!/bin/bash
TRAIN_DATA=${1:-'../data_uORF/1200shape_50bg'}
TEST_DATA=${2:-'../data_uORF/room_diverse_test'}

python train.py --train_dataroot $TRAIN_DATA  --test_dataroot $TRAIN_DATA \
    --n_scenes 1 --n_img_each_scene 4 --display_grad --display_freq 1 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 0 \
    --no_locality_epoch 0 --z_dim 64 --num_slots 5 --bottom \
    --batch_size 1 --num_threads 10 \

# done
echo "Done"