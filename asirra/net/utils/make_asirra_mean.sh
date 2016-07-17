#!/usr/bin/env sh
# Compute the mean image from the asirra training lmdb
# Based on the imagenet script from Caffe

EXAMPLE=/path/to/lmdb/directory
DATA=/path/to/output/directory
TOOLS=/path/to/caffe/tools/directory

$TOOLS/compute_image_mean $EXAMPLE/asirra_train_lmdb \
  $DATA/asirra_mean.binaryproto

echo "Done."
