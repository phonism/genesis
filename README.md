# Genesis
Gensis is a lightweight deep learning framework written from scratch in Python, with Triton as its backend for high-performance computing and Torch for GPU memory management.

## Installation
You can install Otter by cloning the repository and installing the necessary dependencies:
```
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -r requirements.txt
pytest
```

## Usage
Here is a simple example of how to use Otter for a basic neural network:
```
import genesis

# Define a simple model
class SimpleModel(genesis.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = genesis.nn.Linear(10, 5)

    def forward(self, x): 
        return self.fc(x)

# Initialize model and input tensor
model = SimpleModel()
input_tensor = genesis.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

# Forward pass
output = model(input_tensor)
print(output)
```
#### Running the Example
+ Simply run the script using `python sample.py`

## Benchmark
I conducted performance tests for some layers in the benchmark directory, and the results are as follows. A complete benchmark will be added later.
```
# Environment: A100, cuda12.3, torch==2.4.1, triton==3.0.0
# MultiheadAttention
torch                cost_time: 0.3594629764556885
genesis_triton       cost_time: 1.7230677604675293
genesis_fused_triton cost_time: 0.5845096111297607

# LayerNorm
torch                cost_time: 0.17989349365234375
fused_torch          cost_time: 0.020430803298950195
genesis_triton       cost_time: 0.9555635452270508
genesis_fused_triton cost_time: 0.058998823165893555
```

## Current Supported Features
+ Tensor operations: add, mul, matmul, transpose, split, etc.
+ Neural network layers: Linear, ReLU, MultiheadAttention, etc.
+ Automatic differentiation with backpropagation.
+ Basic optimization support: Gradient descent, SGD, AdamW, etc.
+ Integration with Triton for optimized GPU performance.

## Contributing
Feel free to contribute by submitting issues or pull requests. Contributions to expand functionality or improve performance are always welcome!

## License
This project is licensed under the MIT License.
