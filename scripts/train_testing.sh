#!/bin/bash
DATAFILE_HOME=${1:-'../data_uORF/1200shape_50bg.tar.gz'}
DATAROOT="$TMP/data"
DATAFILE="$TMP/data.tar.gz"

if [ -d "$DATAROOT" ]; then
    echo "$DATAROOT already exists."
    echo "Skipping copy - data already copied."
else
    mkdir $DATAROOT
    echo "Copying data from $DATAFILE_HOME to $DATAFILE"
    cp $DATAFILE_HOME $DATAFILE
    echo "Untar file $DATAFILE to $DATAROOT"
    tar -zxf $DATAFILE -C $DATAROOT --strip-components=1
    echo "Finished copying and untar to local disk."
fi



python train.py --train_dataroot $DATAROOT  --test_dataroot "" \
    --n_scenes 5000 --n_img_each_scene 4 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 120 \
    --no_locality_epoch 60 --z_dim 64 --num_slots 5 --bottom \
    --batch_size 1 --num_threads 10 --display_freq 10 \

# done
echo "Done"