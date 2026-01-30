# micrograd.cpp

A C++ version of [Andrej Karpathy's micrograd library](https://github.com/karpathy/micrograd). The goal is to provide a simple, efficient but complete implementation of the micrograd library in C++.

## Features

- **Scalar-valued Autograd Engine**: Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG.
- **Neural Network Library**: High-level modules for building and training neural networks.
- **Header-only**: Easy to integrate into any C++ project; just include the headers.
- **Pytorch-like API**: Familiar interface for users coming from Python/Pytorch.
- **Automatic Memory Management**: Uses `std::shared_ptr` to handle the lifecycle of nodes in the computational graph, ensuring safety and preventing memory leaks.

## Core Engine

The engine implementation (`engine.hpp`) defines the `Value` class, which tracks data and gradients.

### Supported Operations

- **Basic Arithmetic**: `+`, `-`, `*`, `/`, and their assignment variants (`+=`, etc.)
- **Power & Exponentiation**: `pow(value, power)`, `exp(value)`
- **Activation Functions**: `relu()`
- **Negation**: `-value`

## Neural Network Library

The library (`nn.hpp`) provides building blocks for deep learning:

- **Module**: Base class for all NN components (similar to `torch.nn.Module`).
- **Neuron**: A single neuron with weights and a bias.
- **Layer**: A collection of neurons forming a fully connected layer.
- **MLP**: A multi-layer perceptron.

## Installation

Since this is a header-only library, you can simply copy the `include/micrograd` directory to your project.

```cpp
#include <micrograd/engine.hpp>
#include <micrograd/nn.hpp>
```

## Usage

The following example demonstrates building a computational graph and performing backpropagation, matching the canonical example from the original micrograd:

```cpp
#include <micrograd/engine.hpp>
#include <iostream>
#include <iomanip>

int main() {
    Value a(-4.0);
    Value b(2.0);
    Value c = a + b;
    Value d = a * b + pow(b, 3);
    c += c + 1;
    c += 1 + c + (-a);
    d += d * 2 + (b + a).relu();
    d += 3 * d + (b - a).relu();
    Value e = c - d;
    Value f = pow(e, 2);
    Value g = f / 2.0;
    g += 10.0 / f;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << g->data << std::endl; // prints 24.7041
    
    g.backward();

    std::cout << a->grad << std::endl; // prints 138.8338
    std::cout << b->grad << std::endl; // prints 645.5773

    return 0;
}
```

## License

```plaintext
MIT License

Copyright (c) 2026 Siddhesh Agarwal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
