import numpy as np
import activations as act

class FC:
    def __init__(self, in_size, out_size, act):
        self.w = 1/np.sqrt(in_size) * np.random.randn(out_size, in_size)
        self.b = np.random.randn(out_size, 1)
        self.act = act
    
    def feedforward(self, x):
        self.x = x
        self.z = self.w @ self.x + self.b
        self.a = self.act.f(self.z)
        return self.a
    
    def backprop(self, upstream):
        self.delta = self.act.g(self.z) * upstream
        self.dw = self.delta @ self.x.T
        self.db = np.sum(self.delta, axis=1, keepdims=True)
        downstream = self.w.T @ self.delta
        return downstream

      
class NLLSoftmax(FC):    
    def __init__(self, in_size, out_size):
        FC.__init__(self, in_size, out_size, act.Softmax_nograd)   
        
    def backprop(self, a, y):
        upstream = a - y
        return FC.backprop(self, upstream)

    @staticmethod
    def cost(a, y):
        return -np.sum(y * np.log(a))
    

class QuadSigmoid(FC):
    def __init__(self, in_size, out_size):
        FC.__init__(self, in_size, out_size, act.Sigmoid)
        
    def backprop(self, a, y):
        upstream = a - y
        return FC.backprop(self, upstream)

    @staticmethod
    def cost(a, y):
        M = a - y
        return 1/2 * np.sum(M**2)

