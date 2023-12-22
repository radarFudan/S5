# Baseline, 30M model, T=2048, standard epochs
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=0 python train.py -o output_dir_name_v0 -c configs/hyena_S5/wikitext_S5_v0.yaml

# Baseline, 30M model, use the consecutive dataloader, standard epochs
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python train.py -o output_dir_name_v1 -c configs/hyena_S5/wikitext_S5_v1.yaml

# Baseline, 30M model, use the consecutive dataloader, provide the hidden state to the next batch, 
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=2 python train.py -o output_dir_name_v2 -c configs/hyena_S5/wikitext_S5_v2.yaml 

# Baseline, 30M model, use the mamba as the block, provide the hidden state to the next batch,
