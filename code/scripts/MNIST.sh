#!/bin/bash

cd ../

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=$1
Dataset=$2
use_wandb=$3
group_name=$4

if [ -z "$MODEL" ]; then
    MODEL="RNN"
fi

if [ -z "$Dataset" ]; then
    Dataset="sMNIST"
fi

if [ -z "$use_wandb" ]; then
    use_wandb=false
fi

if [ -z "$group_name" ]; then
    group_name="default"
fi

EXTRA_ARGS=()

if [ "$MODEL" == "Transformer" ]; then
    EXTRA_ARGS+=("model_args.num_layers=4")
    EXTRA_ARGS+=("model_args.hidden_size=64")
fi

for SEED in 1 2 3 4 5 6 7 8 9 10;
do
    python main.py seed=$SEED model=$MODEL dataset=$Dataset use_wandb=$use_wandb group_name=$group_name ${EXTRA_ARGS[@]}
done