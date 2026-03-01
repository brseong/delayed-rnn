#!/bin/bash
trap kill SIGINT

indices=(0)
gpus=(4)
models=("DelayedRNN")
hidden_sizes=(180)
# gpus=(4 5 6 7)
# models=("RNN" "LSTM" "GRU" "DelayedRNN")
# hidden_sizes=(256 128 148 180)
max_delay=40
max_think_steps=100
seed=42
batch_size=256
input_size=11
# seq_length=784
seq_min=5
seq_max=100
num_classes=10
learning_rate=0.001
epochs=2000
device="cuda"

for index in "${indices[@]}"; do
    model_type=${models[$((index))]}
    hidden_size=${hidden_sizes[$((index))]}
    # batch_size=${batch_sizes[$((index))]}
    script="CUDA_VISIBLE_DEVICES=${gpus[$((index))]} python qswap.py \
        --model_type ${model_type} \
        --max_delay $max_delay \
        --max_think_steps $max_think_steps \
        --seed $seed \
        --batch_size $batch_size \
        --input_size $input_size \
        --seq_min $seq_min \
        --seq_max $seq_max \
        --hidden_size $hidden_size \
        --num_classes $num_classes \
        --learning_rate $learning_rate \
        --epochs $epochs \
        --device $device"
    echo $script
    eval $script &
done
wait
