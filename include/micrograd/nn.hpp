#include "engine.hpp"
#include <random>
#include <vector>

inline double uniform_rand(double lower, double upper) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(lower, upper);
  return dist(gen);
}

class Module {
  void zero_grad() {
    for (auto &param : parameters()) {
      param->grad = 0.0;
    }
  }

  std::vector<Value> parameters() { return {}; }
};

class Neuron : public Module {
public:
  std::vector<Value> weights;
  Value bias;
  bool nonlin;

  Neuron(int nin, bool nonlin = true)
      : bias(Value(0.0)), nonlin(nonlin) {
    for (unsigned int i = 0; i < nin; i++) {
      weights.push_back(Value(uniform_rand(-1.0, 1.0)));
    }
  }

  Value operator()(std::vector<Value> xs) {
    Value act = bias;
    auto len = xs.size();
    for (unsigned int i = 0; i < len; i++) {
      act += weights[i] * xs[i];
    }
    return (nonlin ? act.relu() : act);
  }

  std::vector<Value> parameters() {
    std::vector<Value> out(weights.size() + 1);
    std::copy(weights.begin(), weights.end(), out.begin());
    out.push_back(bias);
    return out;
  }
};

class Layer : public Module {
public:
  std::vector<Neuron> neurons;

  Layer(int nin, int nout, bool nonlin = true)
      : neurons(nout, Neuron(nin, nonlin)) {}

  std::vector<Value> operator()(std::vector<Value> xs) {
    std::vector<Value> out(neurons.size());
    for (unsigned int i = 0; i < neurons.size(); i++) {
      out[i] = neurons[i](xs);
    }
    return out;
  }

  std::vector<Value> parameters() {
    std::vector<Value> out;
    for (Neuron neuron : neurons) {
      for (Value param : neuron.parameters()) {
        out.push_back(param);
      }
    }
    return out;
  }
};

class MLP : public Module {
public:
  std::vector<Layer> layers;

  MLP(int nin, std::vector<int> nouts) {
    std::vector<int> sz = nouts;
    sz.insert(sz.begin(), nin);
    auto nouts_size = sz.size();
    for (unsigned int i = 0; i < nouts_size; i++) {
      layers.push_back(Layer(sz[i], sz[i + 1], i != (nouts_size - 1)));
    }
  }

  std::vector<Value> operator()(std::vector<Value> &xs) {
    std::vector<Value> out = xs;
    for (Layer layer : layers) {
      out = layer(out);
    }
    return out;
  }

  std::vector<Value> parameters() {
    std::vector<Value> out;
    for (Layer layer : layers) {
      for (Value param : layer.parameters()) {
        out.push_back(param);
      }
    }
    return out;
  }
};
