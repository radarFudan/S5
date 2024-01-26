for index in {4..15}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "The training sequence length is" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python train.py -o Hyena_S5_previous -c configs/hyena_S5/wikitext_S5_v6.yaml --train_length $length
done

for index in {4..15}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "Evaluating the model trained with training sequence length" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.80 CUDA_VISIBLE_DEVICES=1 python evaluate.py -o "Hyena_S5_previous_T${length}" -c configs/hyena_S5/wikitext_S5_v6.yaml
done