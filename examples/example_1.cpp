#include "../include/micrograd/engine.hpp"
#include <iostream>

int main() {
  Value a = Value(-4.0);
  std::cout << a.data << "\n";
  Value b = Value(2.0);
  std::cout << b.data << "\n";
  Value c = a + b;
  std::cout << c.data << "\n";
  Value d = a * b + pow(b, 3);
  std::cout << (a * b).data << " " << pow(b, 3).data << " " << d.data << "\n";
  c += c + 1;
  std::cout << c.data << "\n";
  c += 1 + c + (-a);
  std::cout << c.data << "\n";
  d += d * 2 + (b + a).relu();
  std::cout << d.data << "\n";
  d += 3 * d + (b - a).relu();
  std::cout << d.data << "\n";
  Value e = c - d;
  std::cout << e.data << "\n";
  Value f = pow(e, 2);
  std::cout << f.data << "\n";
  Value g = f / 2.0;
  std::cout << g.data << "\n";
  g += 10.0 / f;
  std::cout << g.data << "\n";
  g.backward();
  std::cout << a.grad << "\n";
  std::cout << b.grad << "\n";
  return 0;
}
