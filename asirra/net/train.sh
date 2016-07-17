#!/usr/bin/env sh

#Change to caffe directory
TOOLS=/home/alex/git/caffe/build/tools

$TOOLS/caffe train \
    --solver=solver.prototxt

