# Genesis
Gensis is a lightweight deep learning framework written from scratch in Python, with Triton as its backend for high-performance computing and Torch for GPU memory management.

## Installation
You can install Otter by cloning the repository and installing the necessary dependencies:
```
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -r requirements.txt
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
+ On GPU: Simply run the script using `python sample.py`
+ On CPU: Set the environment variable to use Torch as the backend for tensor computations by running `NDARRAY_BACKEND=TORCH python sample.py`

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
