#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <vector>

class Value {
public:
  std::function<void()> _backward;
  std::set<Value *> _prev;
  std::string _op;
  double data;
  double grad;

  Value(double data, std::vector<Value *> children = {}, std::string op = "")
      : data(data), grad(0.0), _backward(nullptr),
        _prev(std::set<Value *>(children.begin(), children.end())), _op(op) {}

  // addition
  friend Value operator+(const Value &lhs, const Value &rhs) {
    return Value(lhs.data + rhs.data);
  }

  friend Value operator+(const Value &lhs, double rhs) {
    return Value(lhs.data + rhs);
  }

  friend Value operator+(double lhs, const Value &rhs) {
    return Value(lhs + rhs.data);
  }

  friend Value operator+=(Value &lhs, const Value &rhs) {
    Value out(lhs.data + rhs.data);
    lhs.data = out.data;
    return out;
  }

  friend Value operator+=(Value &lhs, double rhs) {
    Value out(lhs.data + rhs);
    lhs.data = out.data;
    return out;
  }

  // subtraction
  friend Value operator-(const Value &lhs, Value &rhs) { return lhs + (-rhs); }

  friend Value operator-(const Value &lhs, double rhs) { return lhs + (-rhs); }

  friend Value operator-(double lhs, Value &rhs) { return Value(lhs) - rhs; }

  friend Value operator-=(Value &lhs, const Value &rhs) {
    Value out(lhs.data - rhs.data);
    lhs.data = out.data;
    return out;
  }

  friend Value operator-=(Value &lhs, double rhs) {
    Value out(lhs.data - rhs);
    lhs.data = out.data;
    return out;
  }

  // multiplication
  friend Value operator*(Value &lhs, Value &rhs) {
    Value out(lhs.data * rhs.data, {&lhs, &rhs}, "*");
    out._backward = [&]() {
      lhs.grad += out.grad * rhs.data;
      rhs.grad += out.grad * lhs.data;
    };
    return out;
  }

  friend Value operator*(Value &lhs, double rhs) {
    Value other(rhs);
    return lhs * other;
  }

  friend Value operator*(double lhs, Value &rhs) { return rhs * lhs; }

  friend Value operator*=(Value &lhs, const Value &rhs) {
    Value out(lhs.data * rhs.data);
    lhs.data = out.data;
    return out;
  }

  friend Value operator*=(Value &lhs, double rhs) {
    Value out(lhs.data * rhs);
    lhs.data = out.data;
    return out;
  }

  // power
  friend Value pow(Value &lhs, double rhs) {
    Value out(std::pow(lhs.data, rhs), {&lhs}, "**" + std::to_string(rhs));
    out._backward = [&]() {
      lhs.grad += out.grad * rhs * std::pow(lhs.data, rhs - 1);
    };
    return out;
  }

  // ReLU
  Value relu() {
    Value out(std::max(0.0, data), {this}, "ReLU");
    out._backward = [&]() { grad += out.grad * (data > 0); };
    return out;
  }

  // backward
  void backward() {
    std::set<Value *> visited;
    std::vector<Value *> topo;
    this->grad = 1.0;

    std::function<void(Value *)> build_topo = [&](Value *node) {
      if (visited.find(node) != visited.end())
        return;
      visited.insert(node);
      for (const auto &child : node->_prev)
        build_topo(child);
      topo.push_back(node);
    };

    build_topo(this);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      if ((*it)->_backward) {
        (*it)->_backward();
      }
    }
  }

  // negation
  Value operator-() {
    Value out = *this * -1.0;
    return out;
  }

  // division
  friend Value operator/(Value &lhs, Value &rhs) {
    Value out = pow(rhs, -1.0);
    return lhs * out;
  }

  friend Value operator/(Value &lhs, double rhs) {
    Value out(lhs.data / rhs);
    lhs.grad += out.grad * std::pow(rhs, -1);
    return out;
  }

  friend Value operator/(double lhs, Value &rhs) {
    Value temp(lhs);
    return temp / rhs;
  }

  friend Value operator/=(Value &lhs, const Value &rhs) {
    Value out(lhs.data / rhs.data);
    lhs.grad += out.grad * (1.0 / rhs.data);
    return out;
  }

  friend Value operator/=(Value &lhs, double rhs) {
    Value out(lhs.data / rhs);
    lhs.grad += out.grad * (1.0 / rhs);
    return out;
  }
};
