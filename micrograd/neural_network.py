import numpy as np
from micrograd.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

    def parameters(self):
        return []
    

class Neuron(Module):
    def __init__(self, nin, nonLinear = True):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonLinear = nonLinear 

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)),self.b)
        return act.relu() if self.nonLinear else act
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{{'ReLu' if self.nonLinear else 'Linear'}}Neuron({len(self.w)})"
  
    

class Layer(Module):
    def __init__(self,nin,nout):
        self.neuron = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neuron]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neuron for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{','.join(str(n) for n in self.neuron)}]"
    

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layer)}]"
