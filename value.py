from math import exp, e

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.label = label

        self.grad = 0.0
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data}, label={self.label})' if len(self.label) > 0 else f'Value(data={self.data})'

    def __add__(self, other: Value | int | float):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            return
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other: Value | int | float):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            return
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other: int | float):
        if not isinstance(other, (int, float)):
            raise NotImplementedError('Only ints or floats can be powers! Other types are not implemented!')
        out = Value(self.data**other, (self,), f'^{other}')
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
            return
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):        
        return self + (-other)
    
    def tanh(self):
        fx = (exp(2 * self.data) - 1) / (exp(2 * self.data) + 1)
        out = Value(fx, (self,), 'tanh')
        def _backward():
            self.grad += (1 - fx**2) * out.grad
            return
        out._backward = _backward
        return out

    def backward(self):
        nodes = []
        visited = set()
        
        def build_nodes(node):
            if node not in visited:
                visited.add(node)
            for child in node._children:
                build_nodes(child)
            nodes.append(node)
        
        build_nodes(self)
        
        self.grad = 1
        for node in reversed(nodes):
            node._backward()