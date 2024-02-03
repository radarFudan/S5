#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <variable>"
  exit 1
fi

# Assign the first argument to a variable
my_variable="$1"

# Calculate 2 to the power of the index
length=$((2**my_variable))

# echo "The training sequence length is" $length
# XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=0 \
#   /home/aiops/wangsd/miniforge3/envs/S5AIP/bin/python train.py \
#   -o Hyena_S5_previous_full \
#   -c configs/hyena_S5/wikitext_S5_v8.yaml \
#   --train_length $length

echo "Evaluating the model trained with training sequence length" $length
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=0 \
  /home/aiops/wangsd/miniforge3/envs/S5AIP/bin/python evaluate.py \
  -o "Hyena_S5_previous_full_T${length}" \
  -c configs/hyena_S5/wikitext_S5_v8.yaml

