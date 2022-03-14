#!/bin/bash
DATAFILE_HOME=${1:-'../data_uORF/room_diverse_test.tar.gz'}
CHECKPOINT=${2:-'./'}

DATAROOT="$TMP/data"
DATAFILE="$TMP/data.tar.gz"
mkdir $DATAROOT

echo "Copying data from $DATAFILE_HOME to $DATAFILE"
cp $DATAFILE_HOME $DATAFILE
echo "Untar file $DATAFILE to $DATAROOT"
tar -zxf $DATAFILE -C $DATAROOT --strip-components=1
echo "Finished copying and untar to local disk."

python test.py --train_dataroot "" --test_dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoint $CHECKPOINT \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 --gpus 1 \

echo "Done"