for index in {6..13}
do
  length=$((2**index))  # Calculate 2 to the power of index
  echo "The training sequence length is" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.92 CUDA_VISIBLE_DEVICES=0,1 python train.py -o Hyena_GRU_zero_full -c configs/hyena_S5/wikitext_GRU_v7.yaml --train_length $length

  length=$((2**index))  # Calculate 2 to the power of index
  echo "Evaluating the model trained with training sequence length" $length
  XLA_PYTHON_CLIENT_MEM_FRACTION=.92 CUDA_VISIBLE_DEVICES=0,1 python evaluate.py -o "Hyena_GRU_zero_full_T${length}" -c configs/hyena_S5/wikitext_GRU_v7.yaml
done
