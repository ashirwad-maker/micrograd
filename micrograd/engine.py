class Value:
    def __init__(self, value, _children = (), _op=''):
        self.data = value
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda : None 
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad+= 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other),'*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    def __pow__(self, other):
        out = Value(self.data** other,(self,),'**')
        def _backward():
            self.grad+= other*self.data**(other-1) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data<0 else self.data, (self,),"relu")
        def _backward():
            self.grad += (1 if self.data>0 else 0) *out.grad

        self._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


    def __neg__(self):
        return self*-1
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self,other):
        return self * (other**-1)
    
    def __rtruedeiv__(self, other):
        return other * self**-1

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(Data ={self.data} and Grad = {self.grad})"