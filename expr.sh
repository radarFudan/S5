# Baseline, 30M model, T=2048, standard epochs
# Comes with convolution component
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=0 python train.py -o output_dir_name_v0 -c configs/hyena_S5/wikitext_S5_v0.yaml

# Baseline, 30M model, use the consecutive dataloader, standard epochs
# Comes with convolution component
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python train.py -o output_dir_name_v1 -c configs/hyena_S5/wikitext_S5_v1.yaml

# Baseline, 30M model, use the consecutive dataloader, provide the hidden state to the next batch, 
# Pure recurrent
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=2 python train.py -o output_dir_name_v2 -c configs/hyena_S5/wikitext_S5_v2.yaml 
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=2 python evaluate.py -o output_dir_name_v2 -c configs/hyena_S5/wikitext_S5_v2.yaml 

# Baseline, 30M model, use the mamba as the block, provide the hidden state to the next batch,
# Pure recurrent
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=3 python train.py -o output_dir_name_v3 -c configs/hyena_S5/wikitext_S5_v3.yaml 
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=3 python evaluate.py -o output_dir_name_v3 -c configs/hyena_S5/wikitext_S5_v3.yaml 

# For the debug purpose 
# TODO, there is some problem with this evaluation code 
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=0 python evaluate.py -o output_dir_name_v4 -c configs/hyena_S5/wikitext_S5_v4.yaml 
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=.60 CUDA_VISIBLE_DEVICES=3 python evaluate.py -o output_dir_name_v4 -c configs/hyena_S5/wikitext_S5_v4.yaml 

# Want to write the mamba code. 
# Difficulty: It's hard to inherit the speed performance from mamba. 

# Debug the batch training over different training length
# expr1
for index in {4..15}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "The training sequence length is" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=2 python train.py -o Hyena_S5_zero -c configs/hyena_S5/wikitext_S5_v5.yaml --train_length $length
done

# expr2
for index in {4..15}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "The training sequence length is" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=3 python train.py -o Hyena_S5_previous -c configs/hyena_S5/wikitext_S5_v6.yaml --train_length $length
done