for index in {5..7} # 32, 64, 128
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "Evaluating the model trained with training sequence length" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python evaluate.py -o "Hyena_S5_zero_full_T${length}" -c configs/hyena_S5/wikitext_S5_v7.yaml

  length=$((2**index))  # Calculate 2 to the power of index
  echo "Evaluating the model trained with training sequence length" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python evaluate.py -o "Hyena_S5_previous_full_T${length}" -c configs/hyena_S5/wikitext_S5_v8.yaml
done