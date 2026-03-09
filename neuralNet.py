import random

from value import Value

class Neuron:
    def __init__(self, n_in: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x: list[int | float | Value]):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x: list[int | float | Value]):
        outs = [n(x) for n in self.neurons]
        # Please note: The below return will break if topologies converge to 1 node and then diverge!
        return outs if len(outs) > 1 else outs[0]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_in: int, n_outs: tuple[int]):
        self.topology = [n_in] + list(n_outs)
        self.layers = [Layer(self.topology[i], self.topology[i + 1]) for i in range(len(self.topology) - 1)]
    
    def __call__(self, x: list[int | float | Value]):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
