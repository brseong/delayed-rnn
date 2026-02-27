#!/bin/bash
trap kill SIGINT

indices=(0 1 2 3 4 5)
models=("RNN" "LSTM" "DelayedRNN" "RNN" "LSTM" "DelayedRNN")
max_delay=40
max_think_steps=100
seed=42
batch_size=32
input_size=11
# seq_length=784
seq_min=5
seq_max=20
hidden_sizes=(256 128 180 512 256 360)
num_classes=10
learning_rate=0.01
epochs=100

for index in "${indices[@]}"; do
    model_type=${models[$((index))]}
    hidden_size=${hidden_sizes[$((index))]}
    script="CUDA_VISIBLE_DEVICES=${index} python qswap.py \
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
        --epochs $epochs"
    echo $script
    eval $script &
done
wait
