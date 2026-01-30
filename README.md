# micrograd.cpp

A C++ version of [Andrej Karpathy's micrograd library](https://github.com/karpathy/micrograd). The goal is to provide a simple, efficient but complete implementation of the micrograd library in C++.

## Features

The file contains 2 header files:
1. [nn.hpp](./include/nn.hpp): Contains the neural network implementation. Equivalent to the [micrograd/nn.py](https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py) file in the original repository.
2. [engine.hpp](./include/engine.hpp): Contains the engine implementation. Equivalent to the [micrograd/engine.py](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) file in the original repository.

## Usage

To use the library, simply include the header files in your project and link against the library.

```cpp
#include <micrograd/nn.hpp>
#include <micrograd/engine.hpp>
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
