#include "../include/engine.hpp"
#include <iostream>

int main() {
  Value a = Value(-4.0);
  Value b = Value(2.0);
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
  std::cout << g.data << "\n";
  g.backward();
  std::cout << a.grad << "\n";
  std::cout << b.grad << "\n";
  return 0;
}
