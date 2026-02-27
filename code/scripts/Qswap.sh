#!/bin/bash

cd ../

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=$1
use_wandb=$2
group_name=$3

if [ -z "$MODEL" ]; then
    MODEL="RNN"
fi

if [ -z "$use_wandb" ]; then
    use_wandb=false
fi

if [ -z "$group_name" ]; then
    group_name="default"
fi

EXTRA_ARGS=()
data_name="Qswap"

num_epochs=50
hidden_size=256
min_seq_len=5
max_seq_len=50

if [ "$MODEL" == "Transformer" ]; then
    EXTRA_ARGS+=("model_args.num_layers=1")
    EXTRA_ARGS+=("model_args.hidden_size=128")
fi

for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19;
do
    ARGS=(
        "num_workers=0"
        "seed=$SEED"  
        "model=$MODEL"
        "dataset=$data_name"
        "use_wandb=$use_wandb"
        "wandb.group_name=$group_name"
        "num_epochs=$num_epochs"
        "model_args.hidden_size=$hidden_size"
        "dataset.min_seq_len=$min_seq_len"
        "dataset.max_seq_len=$max_seq_len"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done