#!/bin/bash
trap kill SIGINT

indices=(0)
gpus=(0)
models=("DelayedRNN")
hidden_sizes=(180)
# gpus=(4 5 6 7)
# models=("RNN" "LSTM" "GRU" "DelayedRNN")
# hidden_sizes=(256 128 148 180)
max_delay=40
# max_think_steps=100
seed=42
batch_size=256
input_size=1
seq_length=784
# seq_min=5
# seq_max=100
num_classes=10
learning_rate=0.003
epochs=2000
device="cuda"

for index in "${indices[@]}"; do
    model_type=${models[$((index))]}
    hidden_size=${hidden_sizes[$((index))]}
    # batch_size=${batch_sizes[$((index))]}
    # learning_rate=${learning_rates[$((index))]}
    script="CUDA_VISIBLE_DEVICES=${gpus[$((index))]} python psmnist.py \
        --model_type ${model_type} \
        --max_delay $max_delay \
        --seed $seed \
        --batch_size $batch_size \
        --input_size $input_size \
        --seq_length $seq_length \
        --hidden_size $hidden_size \
        --num_classes $num_classes \
        --learning_rate $learning_rate \
        --epochs $epochs \
        --device $device"
    echo $script
    eval $script &
done
wait
