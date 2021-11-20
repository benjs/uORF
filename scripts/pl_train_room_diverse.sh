#!/bin/bash
DATAROOT=${1:-'../data_uORF/1200shape_50bg'}
PORT=${2:-6066}
python -m visdom.server -p $PORT &>/dev/null &
python pl_train.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 120 --no_locality_epoch 60 --z_dim 64 --num_slots 5 \
    --model 'uorf_gan' --bottom --batch_size 1 \
# done
echo "Done"
