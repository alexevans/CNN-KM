#!/usr/bin/env sh
# Create the asirra lmdb inputs
# Based on the imagenet script from Caffe
# N.B. set the path to the asirra train + val data dirs

EXAMPLE=/path/to/output/directory/
DATA=/path/to/dataset/directory/
TOOLS=/path/to/caffe/tools/directory/

TRAIN_DATA_ROOT=/home/alex/datasets/dogcat/train/
VAL_DATA_ROOT=/home/alex/datasets/dogcat/val/

#some images are not actually jpgs even though they have the .jpg extension
mogrify -format jpg $TRAIN_DATA_ROOT/*.jpg
mogrify -format jpg $VAL_DATA_ROOT/*.jpg

# Set RESIZE=true to resize the images to 227x227. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/asirra_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/asirra_val_lmdb

echo "Done."
