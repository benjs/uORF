#!/bin/bash
DATAFILE_HOME=${1:-'../data_uORF/1200shape_50bg.tar.gz'}
DATAROOT="$TMP/mydata"
DATAFILE="$TMP/data.tar.gz"

mkdir $DATAROOT
echo "Copying data from $DATAFILE_HOME to $DATAFILE"
cp $DATAFILE_HOME $DATAFILE
echo "Untar file $DATAFILE to $DATAROOT"
tar -zxf $DATAFILE -C $DATAROOT --strip-components=1
echo "Finished copying and untar to local disk."

python train.py --train_dataroot $DATAROOT  --test_dataroot "" \
    --n_scenes 5000 --n_img_each_scene 4 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 120 \
    --no_locality_epoch 60 --z_dim 64 --num_slots 5 --bottom --attn_iter 4 \
    --batch_size 1 --num_threads 10 --display_freq 1000 \

# done
echo "Done"