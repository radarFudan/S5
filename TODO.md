1. Another dataloader
    Fnished, running in v1, **TODO**, still need a sanity check code to verify the consecutive correctness. 

2. Evaluate over two dataloaders
    Write the evaluate code, output the final result into a csv file with train sequence length in the name. 
    Use `evaluate.py`, **running** in v4, 

3. Make the model take the previous hidden states
    Finished, running in v2/v3, 

4. Brush across different training sequence length from 16 to 32k. 
    Then we can make a table for 30M model. 

5. Implement a JAX mamba code based on the simplified version, but this might have the efficiency issue. 
    **TODO** Write the hidden state update formula first, both the discrete and continuous time version
    Based on [Mamba-minimal](https://github.com/johnma2006/mamba-minimal)

6. Explore the context extension performance from previous hidden states, done, it is monotonic

7. Evaluate the hidden state and per token perplexity of the validation and test dataset. 
