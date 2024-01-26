
for index in {4..4}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "Evaluating the model trained with training sequence length" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=3 python evaluate.py -o "Hyena_Mamba_zero_T${length}" -c configs/hyena_S5/wikitext_Mamba_v5.yaml
done
