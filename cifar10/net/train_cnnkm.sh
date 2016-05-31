#!/usr/bin/env sh

#Change to caffe directory
TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=cifar10_cnnkm_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=cifar10_cnnkm_solver_lr1.prototxt \
    --snapshot=cifar10_cnnkm_iter_60000.solverstate

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=cifar10_cnnkm_solver_lr2.prototxt \
    --snapshot=cifar10_cnnkm_iter_65000.solverstate
