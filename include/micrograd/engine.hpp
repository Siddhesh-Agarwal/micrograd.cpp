#ifndef MICROGRAD_ENGINE_HPP
#define MICROGRAD_ENGINE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <iostream>

struct ValueImpl {
  double data;
  double grad;
  std::string _op;
  std::vector<std::shared_ptr<ValueImpl>> _prev;
  std::function<void(ValueImpl *)> _backward;

  ValueImpl(double data, std::vector<std::shared_ptr<ValueImpl>> children = {},
            std::string op = "")
      : data(data), grad(0.0), _prev(children), _op(op) {}
};

class Value {
  std::shared_ptr<ValueImpl> impl;

public:
  Value(double data) : impl(std::make_shared<ValueImpl>(data)) {}
  Value() : Value(0.0) {}
  Value(std::shared_ptr<ValueImpl> ptr) : impl(ptr) {}

  ValueImpl *operator->() { return impl.get(); }
  const ValueImpl *operator->() const { return impl.get(); }

  // Access to internal pointer for graph building
  std::shared_ptr<ValueImpl> get_impl() const { return impl; }

  // addition
  friend Value operator+(const Value &lhs, const Value &rhs) {
    Value out(std::make_shared<ValueImpl>(
        lhs->data + rhs->data,
        std::vector<std::shared_ptr<ValueImpl>>{lhs.impl, rhs.impl}, "+"));
    
    out.impl->_backward = [l = lhs.impl, r = rhs.impl](ValueImpl *out_node) {
      l->grad += out_node->grad;
      r->grad += out_node->grad;
    };
    return out;
  }

  friend Value operator+(const Value &lhs, double rhs) {
    Value out(std::make_shared<ValueImpl>(
        lhs->data + rhs,
        std::vector<std::shared_ptr<ValueImpl>>{lhs.impl}, "+"));
        
    out.impl->_backward = [l = lhs.impl](ValueImpl *out_node) {
      l->grad += out_node->grad;
    };
    return out;
  }

  friend Value operator+(double lhs, const Value &rhs) { return rhs + lhs; }

  friend Value operator+=(Value &lhs, const Value &rhs) {
    lhs = lhs + rhs;
    return lhs;
  }

  friend Value operator+=(Value &lhs, double rhs) {
    lhs = lhs + rhs;
    return lhs;
  }

  // subtraction
  friend Value operator-(const Value &lhs, const Value &rhs) {
    return lhs + (-rhs);
  }

  friend Value operator-(const Value &lhs, double rhs) { return lhs + (-rhs); }

  friend Value operator-(double lhs, const Value &rhs) { return Value(lhs) - rhs; }

  friend Value operator-=(Value &lhs, const Value &rhs) {
    lhs = lhs - rhs;
    return lhs;
  }

  friend Value operator-=(Value &lhs, double rhs) {
    lhs = lhs - rhs;
    return lhs;
  }

  // multiplication
  friend Value operator*(const Value &lhs, const Value &rhs) {
    Value out(std::make_shared<ValueImpl>(
        lhs->data * rhs->data,
        std::vector<std::shared_ptr<ValueImpl>>{lhs.impl, rhs.impl}, "*"));
        
    out.impl->_backward = [l = lhs.impl, r = rhs.impl](ValueImpl *out_node) {
      l->grad += out_node->grad * r->data;
      r->grad += out_node->grad * l->data;
    };
    return out;
  }

  friend Value operator*(const Value &lhs, double rhs) {
    return lhs * Value(rhs);
  }

  friend Value operator*(double lhs, const Value &rhs) { return rhs * lhs; }

  friend Value operator*=(Value &lhs, const Value &rhs) {
    lhs = lhs * rhs;
    return lhs;
  }

  friend Value operator*=(Value &lhs, double rhs) {
    lhs = lhs * rhs;
    return lhs;
  }

  // division
  friend Value operator/(const Value &lhs, const Value &rhs) {
    return lhs * pow(rhs, -1);
  }

  friend Value operator/(const Value &lhs, double rhs) {
    return lhs * std::pow(rhs, -1);
  }

  friend Value operator/(double lhs, const Value &rhs) {
    return Value(lhs) / rhs;
  }
  
  friend Value operator/=(Value &lhs, const Value &rhs) {
      lhs = lhs / rhs;
      return lhs;
  }

  friend Value operator/=(Value &lhs, double rhs) {
      lhs = lhs / rhs;
      return lhs;
  }

  // power
  friend Value pow(const Value &lhs, double rhs) {
    Value out(std::make_shared<ValueImpl>(
        std::pow(lhs->data, rhs),
        std::vector<std::shared_ptr<ValueImpl>>{lhs.impl},
        "**" + std::to_string(rhs)));
        
    out.impl->_backward = [l = lhs.impl, rhs](ValueImpl *out_node) {
      l->grad += out_node->grad * rhs * std::pow(l->data, rhs - 1);
    };
    return out;
  }
  
  // exponentiation
  friend Value exp(const Value &lhs) {
    Value out(std::make_shared<ValueImpl>(
        std::exp(lhs->data),
        std::vector<std::shared_ptr<ValueImpl>>{lhs.impl}, "exp"));
        
    out.impl->_backward = [l = lhs.impl](ValueImpl *out_node) {
      l->grad += out_node->grad * out_node->data; // derivative of exp(x) is exp(x) (which is out->data)
    };
    return out;
  }

  // ReLU
  Value relu() const {
    Value out(std::make_shared<ValueImpl>(
        std::max(0.0, impl->data),
        std::vector<std::shared_ptr<ValueImpl>>{impl}, "ReLU"));
        
    out.impl->_backward = [ptr = impl](ValueImpl *out_node) {
      ptr->grad += out_node->grad * (out_node->data > 0);
    };
    return out;
  }

  // negation
  Value operator-() const { return *this * -1.0; }

  // backward
  void backward() {
    std::set<ValueImpl *> visited;
    std::vector<ValueImpl *> topo;
    
    std::function<void(ValueImpl *)> build_topo = [&](ValueImpl *node) {
      if (visited.find(node) != visited.end())
        return;
      visited.insert(node);
      for (const auto &child : node->_prev)
        build_topo(child.get());
      topo.push_back(node);
    };

    build_topo(impl.get());
    
    impl->grad = 1.0;
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      if ((*it)->_backward) {
        (*it)->_backward(*it);
      }
    }
  }
};

#endif
