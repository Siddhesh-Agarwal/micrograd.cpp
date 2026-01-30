#include <micrograd/engine.hpp>
#include <iostream>

int main() {
  Value a = Value(-4.0);
  std::cout << "a: " << a->data << "\n";
  Value b = Value(2.0);
  std::cout << "b: " << b->data << "\n";
  Value c = a + b;
  std::cout << "c (a+b): " << c->data << "\n";
  Value d = a * b + pow(b, 3);
  std::cout << "d (a*b + b^3): " << d->data << "\n";
  c += c + 1;
  std::cout << "c (+= c+1): " << c->data << "\n";
  c += 1 + c + (-a);
  std::cout << "c (+= ...): " << c->data << "\n";
  d += d * 2 + (b + a).relu();
  std::cout << "d updated: " << d->data << "\n";
  d += 3 * d + (b - a).relu();
  std::cout << "d updated 2: " << d->data << "\n";
  Value e = c - d;
  std::cout << "e (c-d): " << e->data << "\n";
  Value f = pow(e, 2);
  std::cout << "f (e^2): " << f->data << "\n";
  Value g = f / 2.0;
  std::cout << "g (f/2): " << g->data << "\n";
  g += 10.0 / f;
  std::cout << "g (+= 10/f): " << g->data << "\n";
  g.backward();
  std::cout << "a.grad: " << a->grad << "\n";
  std::cout << "b.grad: " << b->grad << "\n";
  return 0;
}
