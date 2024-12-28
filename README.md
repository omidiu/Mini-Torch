## 🔥 MiniTorch 🔥

<div style="text-align: center;">
    <img src="./statics/pytorch_minitorch.png" alt="Pytorch vs minitorch" width="100%" />
</div>

**MiniTorch** is a minimalist educational library built to uncover the mechanics behind PyTorch, using only Python and
NumPy. This project is ideal for learners and educators who want to deepen their
understanding of how deep learning frameworks work under the hood.

Currently, MiniTorch supports building simple Multi-Layer Perceptrons (MLPs), with plans to expand its functionality.

--------------------------------------------------------------------------------

## ✨ micrograd VS MiniTorch
<div style="text-align: center;">
    <img src="./statics/micrograd_minitorch.png" alt="micrograd vs MiniTorch" width="100%" />
</div>

We began with Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and extended it to support high-dimensional tensor operations, aligning more closely with PyTorch.

- The key difference is that we extended derivatives to work with high-dimensional tensors using matrix calculus. In this perspective, derivatives are linear transformations, like Jacobians, mapping changes in inputs to outputs. This supports operations like matrix multiplication, broadcasting, and reductions while maintaining efficient backpropagation.  For more information, we highly recommend the MIT course *"[Matrix Calculus for Machine Learning and Beyond](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/video_galleries/lecture-videos/)*". It covers most of the mathematical prerequisites. You can watch it or wait for our videos on the topic. 🎥


- Our library is more modular, object-oriented, and closely aligned with the PyTorch API. For example, by inheriting the `Module` class, any custom class can function as a neural network and automatically retrieve all its parameters. This eliminates the need to manually define a `parameters()` method for each module, as is required in micrograd.


**Note**: If we develop this library further, we aim to refactor the `_backward` method. This change will simplify the implementation of `requires_grad`.

--------------------------------------------------------------------------------

## ⭐️ Join Us on Our Journey!

If MiniTorch receives **1k GitHub stars ⭐️**, we will:

1. **Expand the library by adding new features**: We'll implement new `nn.Modules` and extend its functionality. Please
   read the "Future Plans" section in this file for more information about the planned modules.
2. **Create a YouTube playlist <img src="./statics/Youtube_logo.png" alt="YouTube Logo" width="25" />**: We will create a YouTube playlist that teaches you all the knowledge necessary to build this library. It will cover concepts such as:  
- The required mathematics, for example, the concept of a derivative based on a matrix  
- Advanced operations with multi-dimensional tensors  
- Object-oriented concepts, and how PyTorch automatically identifies all the weights in your network  
... and more.

We plan to explain every single line of code written in this repository, enabling you to implement it yourself with sufficient time and patience, and gain a deeper understanding of libraries like PyTorch. Furthermore, if this library continues to develop, we will also teach all the newly added parts. For example, we will learn how the process behind the `TransformerEncoderLayer` works. (*This playlist will be presented in English or Persian.*)




Your support will validate the value of this project and help us provide free, high-quality educational resources for
the AI community.

--------------------------------------------------------------------------------

## 🚀 Quickstart
1- clone the repository:

```bash
git clone https://github.com/omidiu/Mini-Torch.git
```

2- Navigate to the project directory:

```bash
cd minitorch
```

3- Create a virtual environment:

```bash
# For Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate
```

4- Install the required dependency:
```bash
pip install numpy
```

5- Open and run the experiments.ipynb file.

### 1. Build a Simple MLP

```python
from nn import Linear, Module, MSELoss
from optim import Adam
from tensor import Tensor
import nn.functional as F
from nn import Linear as MiniLinear, Module as MiniModule, MSELoss as MiniMSELoss
from optim import Adam as MiniAdam
from tensor import Tensor as MiniTensor
import nn.functional as MiniF

class MLP(MiniModule):
    def __init__(self):
        super().__init__()
        self.linear_1 = MiniLinear(3, 3)
        self.linear_2 = MiniLinear(3, 6)
        self.linear_3 = MiniLinear(6, 1)

    def forward(self, x):
        x = MiniF.tanh(self.linear_1(x))
        x = MiniF.tanh(self.linear_2(x))
        x = MiniF.tanh(self.linear_3(x))
        return x
    
model = MLP()
optim = MiniAdam(model.parameters())
criterion = MiniMSELoss()


X = MiniTensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
Y = MiniTensor([[1.0], [-1.0], [-1.0], [1.0]])

epochs = 100

for epoch in range(epochs):
    y_hat = model(X)
    loss = criterion(y_hat, Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Epoch {epoch}, Loss: {loss.data}")
    
print('\nModel architecture: \n',model)
print('\n state dict:\n' ,model.state_dict())
```

### 2. MiniTorch vs. PyTorch: High Dimensional Tensors
```python
import numpy as np
# torch imports
import torch
import torch.nn.functional as TorchF
from torch import Tensor as TorchTensor

# mini_torch imports
from tensor import Tensor as MiniTensor
import nn.functional as MiniF

def gradients_are_equal(torch_tensor: TorchTensor, mini_tensor: MiniTensor):
    print(np.all(torch_tensor.grad.detach().numpy()==mini_tensor.grad))

val_a = [
    [
        [
            [1, 2, 3, 4]
        ]
    ],
    [
        [
            [1, 9, -1, 4]
        ]
    ],
    [
        [
            [1, 2, 3, -1]
        ]
    ]
]

val_b = [
    [1,   9,   3,   4],
    [0,   1,  -1, -11],
    [1,  21,  11,  -1]
]

a_torch = torch.tensor(val_a, requires_grad=True, dtype=torch.float32)
b_torch = torch.tensor(val_b, requires_grad=True, dtype=torch.float32)
c_torch = TorchF.linear(a_torch, b_torch)
d_torch = c_torch.sum(); d_torch.retain_grad()
d_torch.backward()

a_mini = MiniTensor(val_a)
b_mini = MiniTensor(val_b)
c_mini = MiniF.linear(a_mini, b_mini)
d_mini = c_mini.sum()
d_mini.backward()

gradients_are_equal(a_torch, a_mini) # compare d_d/d_a
gradients_are_equal(b_torch, b_mini) # compare d_d/d_b
```

--------------------------------------------------------------------------------


## 📈 Future Plans

As mentioned above, we will expand the library by adding new features if we reach **1k GitHub stars ⭐️**, along with
creating detailed videos for each feature in a YouTube playlist.

We will implement these modules:

- Modules with a ✅ are already implemented (though they are currently minimal).
- Modules with a `*` are prioritized for implementation in the next updates.

The below list comes from PyTorch's [original documentation](https://pytorch.org/docs/main/nn.html). Some modules are
intentionally omitted as they are out of the scope of this project.

### General

- [ ] `Parameter *`: A type of Tensor with `requires_grad = True`. Note: We haven't implemented requires_grad
  functionality yet.

### Containers

- ✅ `Module`: Base class for all neural network modules.
    - ✅ Parameter registration
    - ✅ Submodule management
    - ✅ Forward method abstraction
    - ✅ Parameter retrieval (with support for submodules)
- [ ] `Sequential *`: A sequential container.
- [ ] `ModuleList *`: Holds submodules in a list.
- [ ] `ModuleDict *`: Holds submodules in a dictionary.
- [ ] `ParameterList *`: Holds parameters in a list.
- [ ] `ParameterDict *`: Holds parameters in a dictionary.

### Convolution Layers

- [ ] `Conv1d *`: Applies a 1D convolution over an input signal composed of several input planes.
- [ ] `Conv2d *`: Applies a 2D convolution over an input signal composed of several input planes.
- [ ] `Conv3d *`: Applies a 3D convolution over an input signal composed of several input planes.
- [ ] `ConvTranspose1d`: Applies a 1D transposed convolution operator over an input image composed of several input
  planes.
- [ ] `ConvTranspose2d`: Applies a 2D transposed convolution operator over an input image composed of several input
  planes.
- [ ] `ConvTranspose3d`: Applies a 3D transposed convolution operator over an input image composed of several input
  planes.

### Pooling Layers

- [ ] `MaxPool1d *`: Applies a 1D max pooling over an input signal composed of several input planes.
- [ ] `MaxPool2d *`: Applies a 2D max pooling over an input signal composed of several input planes.
- [ ] `MaxPool3d *`: Applies a 3D max pooling over an input signal composed of several input planes.
- [ ] `AvgPool1d *`: Applies a 1D average pooling over an input signal composed of several input planes.
- [ ] `AvgPool2d *`: Applies a 2D average pooling over an input signal composed of several input planes.
- [ ] `AvgPool3d *`: Applies a 3D average pooling over an input signal composed of several input planes.

### Padding Layers

- [ ] `ZeroPad1d *`: Pads the input tensor boundaries with zero.
- [ ] `ZeroPad2d *`: Pads the input tensor boundaries with zero.
- [ ] `ZeroPad3d *`: Pads the input tensor boundaries with zero.

### Non-linear Activations

- ✅ `Tanh *`: Applies the Hyperbolic Tangent (Tanh) function element-wise.
- ✅ `ReLU *`: Applies the rectified linear unit function element-wise.
- ✅ `LeakyReLU *`: Applies the LeakyReLU function element-wise.
- ✅ `Sigmoid *`: Applies the Sigmoid function element-wise.
- [ ] `Softmax *`: Applies the Softmax function to an n-dimensional input Tensor.
- [ ] `SiLU *`: Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

### Normalization Layers

- [ ] `BatchNorm1d *`: Applies Batch Normalization over a 2D or 3D input.
- [ ] `BatchNorm2d *`: Applies Batch Normalization over a 4D input.
- [ ] `BatchNorm3d *`: Applies Batch Normalization over a 5D input.
- [ ] `LayerNorm *`: Applies Layer Normalization over a mini-batch of inputs.
- [ ] `GroupNorm`: Applies Group Normalization over a mini-batch of inputs.

### Recurrent Layers

- [ ] `RNN *`: Simple Recurrent Neural Network.
- [ ] `LSTM *`: Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
- [ ] `GRU *`: Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
- [ ] `RNNCell`: An Elman RNN cell with tanh or ReLU non-linearity.
- [ ] `LSTMCell`: A long short-term memory (LSTM) cell.
- [ ] `GRUCell`: A gated recurrent unit (GRU) cell.

### Transformer Layers

- [ ] `Transformer`: A transformer model.
- [ ] `TransformerEncoder`: TransformerEncoder is a stack of N encoder layers.
- [ ] `TransformerDecoder`: TransformerDecoder is a stack of N decoder layers.
- [ ] `TransformerEncoderLayer`: TransformerEncoderLayer is made up of self-attn and feedforward network.
- [ ] `TransformerDecoderLayer`: TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward
  network.

### Linear Layers

- ✅ `Linear *`: Applies an affine linear transformation to the incoming data:
  \( y = x A^T + b \).

### Dropout Layers

- [ ] `Dropout`: During training, randomly zeroes some of the elements of the input tensor with probability p.
- [ ] `Dropout1d`: Randomly zero out entire channels.
- [ ] `Dropout2d`: Randomly zero out entire channels.
- [ ] `Dropout3d`: Randomly zero out entire channels.

### Distance Functions

- [ ] `CosineSimilarity *`: Returns cosine similarity between `x_1` and `x_2`, computed along `dim`.
- [ ] `PairwiseDistance *`: Computes the pairwise distance between input vectors, or between columns of input matrices.

### Loss Functions

- [ ] `MSELoss *`
- [ ] `CrossEntropyLoss *`
- [ ] `BCELoss *`
- [ ] `KLDivLoss`
- [ ] `TripletMarginLoss`

### Vision Layers

- [ ] `PixelShuffle`: Rearrange elements in a tensor according to an upscaling factor.
- [ ] `PixelUnshuffle`: Reverse the PixelShuffle operation.
- [ ] `Upsample`: Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

### Currently Out of Scope

Hooks, Sparse Layers, Shuffle Layers, DataParallel Layers, Utilities





