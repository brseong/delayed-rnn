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

num_epochs=2000
batch_size=256
hidden_size=256
min_seq_len=5
max_seq_len=100
use_lr_scheduler=false

if [ "$MODEL" == "RNN" ]; then
    EXTRA_ARGS+=("hidden_size=402")
fi

if [ "$MODEL" == "DelayRNN" ]; then
    EXTRA_ARGS+=("hidden_size=256")
    EXTRA_ARGS+=("model.max_delay=50")
fi

if [ "$MODEL" == "GRU" ];  then
    EXTRA_ARGS+=("hidden_size=231")
fi

# GRU와 LSTM 모두 hidden_size를 512로 설정
if [ "$MODEL" == "LSTM" ];  then
    EXTRA_ARGS+=("hidden_size=200")
fi


for SEED in 0 1 2 4 5;
do
    ARGS=(
        "seed=$SEED"  
        "model=$MODEL"
        "dataset=$data_name"
        "wandb.use_wandb=$use_wandb"
        "wandb.group_name=$group_name"
        "num_epochs=$num_epochs"
        "batch_size=$batch_size"
        "use_lr_scheduler=$use_lr_scheduler"
        "dataset.min_seq_len=$min_seq_len"
        "dataset.max_seq_len=$max_seq_len"
        "${EXTRA_ARGS[@]}"
    )
    # > output.log 2>&1
    python main.py "${ARGS[@]}"
done