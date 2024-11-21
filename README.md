## 🔥 MiniTorch 🔥

<div style="text-align: center;">
    <img src="./statics/pytorch_minitorch_.png" alt="YouTube Logo" width="100%" />
</div>

**MiniTorch** is a minimalist educational library built to uncover the mechanics behind PyTorch. Using only Python and
NumPy. This project is ideal for learners and educators who want to deepen their
understanding of how deep learning frameworks work under the hood.

Currently, MiniTorch supports building simple Multi-Layer Perceptrons (MLPs), with plans to expand its functionality.

--------------------------------------------------------------------------------

## ⭐️ Join Us on Our Journey!

If MiniTorch receives **1k GitHub stars ⭐️**, we will:

1. **Expand the library by adding new features**: We'll implement new `nn.Modules` and extend its functionality. Please
   read the "Future Plans" section in this file for more information about the planned modules.
2. **Create a YouTube playlist <img src="./statics/Youtube_logo.png" alt="YouTube Logo" width="25" />**: This series will explore concepts ranging from mathematical prerequisites to advanced
   computer science algorithms, enhanced by sophisticated animations created
   using [Manim](https://github.com/ManimCommunity/manim).
   
<div style="text-align: center;">
    <img src="./statics/Manim_logo.png" alt="YouTube Logo" width="200" />
</div>



Your support will validate the value of this project and help us provide free, high-quality educational resources for
the AI community.

--------------------------------------------------------------------------------

## 🚀 Quickstart

### ⚙️ Installation


First, clone the repository:

```bash
git clone https://github.com/omidiu/minitorch.git
```

Navigate to the project directory:

```bash
cd minitorch
```

Install the only required dependency:

```bash
pip install numpy
```

### 1. Build a Simple MLP

```python
from nn import Linear, Module, MSELoss
from optim import Adam
from tensor import Tensor
import nn.functional as F


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(3, 3)
        self.linear_2 = Linear(3, 6)
        self.linear_3 = Linear(6, 1)

    def forward(self, x):
        x = F.tanh(self.linear_1(x))
        x = F.tanh(self.linear_2(x))
        x = F.tanh(self.linear_3(x))
        return x

model = MLP()
optim = Adam(model.parameters())
criterion = MSELoss()


X = Tensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
Y = Tensor([[1.0], [-1.0], [-1.0], [1.0]])

epochs = 100

for epoch in range(epochs):
    y_hat = model(X)
    loss = criterion(y_hat, Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Epoch {epoch}, Loss: {loss.data}")
```

### 2. MiniTorch vs. PyTorch: Validation
```python
import numpy as np

import torch
import torch.nn as nn_torch
import torch.nn.functional as F_torch

from tensor import Tensor
import nn as nn_mini
import nn.functional as F_mini


def arrays_are_close(arr_1, arr_2):
    return np.allclose(arr_1, arr_2, atol=1e-9)


mse_loss_torch = nn_torch.MSELoss(reduction='sum')
mse_loss_mini = nn_mini.MSELoss(reduction='sum')


a_torch = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float32)
b_torch = torch.tensor([[5, 6], [7, 8]], requires_grad=True, dtype=torch.float32)
y_torch = torch.tensor([[9, 2], [3, -1]], requires_grad=True, dtype=torch.float32)

y_hat_torch = F_torch.linear(a_torch, b_torch)
loss_torch = mse_loss_torch(y_hat_torch, y_torch)
loss_torch.backward()

a_mini = Tensor([[1, 2], [3, 4]])
b_mini = Tensor([[5, 6], [7, 8]])
y_mini = Tensor([[9, 2], [3, -1]])

y_hat_mini = F_mini.linear(a_mini, b_mini)
loss_mini = mse_loss_mini(y_hat_mini, y_mini)
loss_mini.backward()


print(arrays_are_close(a_torch.grad.detach().numpy(), a_mini.grad))
print(arrays_are_close(b_torch.grad.detach().numpy(), b_mini.grad))
```

--------------------------------------------------------------------------------

## ✨ micrograd VS MiniTorch
<div style="text-align: center;">
    <img src="./statics/micrograd_minitorch.png" alt="YouTube Logo" width="100%" />
</div>

We began with Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and extended it to support high-dimensional tensor operations, aligning more closely with PyTorch.

- The key difference is that we extended derivatives to work with high-dimensional tensors using matrix calculus. In this view, derivatives are linear transformations, like Jacobians, mapping changes in inputs to outputs. This supports operations like matrix multiplication, broadcasting, and reductions while maintaining efficient backpropagation.  For more information, we highly recommend the MIT course *"[Matrix Calculus for Machine Learning and Beyond](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/video_galleries/lecture-videos/)*". It covers most of the mathematical prerequisites. You can watch it or wait for our videos on the topic. 🎥


- Our library is more modular, object-oriented, and closely aligned with the PyTorch API. For example, by inheriting the `Module` class, any custom class can function as a neural network and automatically retrieve all its parameters. This eliminates the need to manually define a `parameters()` method for each module, as is required in micrograd.


**Note**: If we develop this library further, we aim to refactor the `_backward` method to make it a property of each tensor (like `grad_fn`) rather than relying on its parents. This change will simplify the implementation of `requires_grad`.

--------------------------------------------------------------------------------
 


## 📈 Future Plans

As mentioned above, we will expand the library by adding new features if we reach **500 GitHub stars ⭐️**, along with
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
- [ ] `ReLU *`: Applies the rectified linear unit function element-wise.
- [ ] `LeakyReLU *`: Applies the LeakyReLU function element-wise.
- [ ] `Sigmoid *`: Applies the Sigmoid function element-wise.
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





