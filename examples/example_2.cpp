#include "../include/micrograd/nn.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create a simple neural network
    auto n1 = Neuron(1.0);
    auto n2 = Neuron(2.0);
    auto n3 = Neuron(3.0);

    std::vector<Value> x = {Value(1.0), Value(2.0), Value(3.0)};
    auto y1 = n1(x);
    auto y2 = n2(x);
    auto y3 = n3(x);

    std::cout << "y1.data: " << y1.data << std::endl;
    std::cout << "y2.data: " << y2.data << std::endl;
    std::cout << "y3.data: " << y3.data << std::endl;

    return 0;
}
