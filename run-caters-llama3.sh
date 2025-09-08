#!/bin/bash

# conda activate temporal_llms


# model_path="meta-llama/Llama-3.1-8B-Instruct"
model_name="meta-llama/Llama-3.1-8B-Instruct"

num_examples=3

export CUDA_VISIBLE_DEVICES=7

nohup python3 caters-llama3.py \
    --model_name $model_name \
    --output_path "llama-output/caters/caters-fs-pt${prompt}-output-icl${num_examples}-del" \
    --temperature 0.8 \
    --top_p 0.95 \
    --do_sample \
    --max_events_length 3696 \
    --max_new_decoding_tokens 128 \
    --caters_eval \
    --do_in_context_learning \
    --max_batch_size 1 \
    --num_example ${num_examples} \
    --train_set ./dataset/caters/caters_train.csv \
    --prompt_template 1 &

# for prompt in 1 2 3
# do
#     # # # ### run mctaco with ICL
#     python3 caters-llama.py \
#         --model_name $model_name \
#         --model_path $model_path \
#         --output_path "llama-output/caters/caters-fs-pt${prompt}-output-icl${num_examples}-del" \
#         --temperature 0.8 \
#         --top_p 0.95 \
#         --do_sample \
#         --max_events_length 3696 \
#         --max_new_decoding_tokens 128 \
#         --caters_eval \
#         --do_in_context_learning \
#         --max_batch_size 1 \
#         --num_example ${num_examples} \
#         --train_set ./dataset/caters/caters_train.csv \
#         --prompt_template ${prompt}
# done
    