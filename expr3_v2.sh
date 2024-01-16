for index in {4..4}
do
  length=$((2**index))  # Calculate 2 to the power of index

  echo "The training sequence length is" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=0 /home/aiops/wangsd/miniforge3/envs/S5AIP/bin/python train.py -o Hyena_Mamba_zero_full -c configs/hyena_S5/wikitext_Mamba_v7_120m.yaml --train_length $length

  # echo "Evaluating the model trained with training sequence length" $length
  # XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=0 /home/aiops/wangsd/miniforge3/envs/S5AIP/bin/python evaluate.py -o "Hyena_Mamba_zero_full_T${length}" -c configs/hyena_S5/wikitext_Mamba_v7_120m.yaml
done
