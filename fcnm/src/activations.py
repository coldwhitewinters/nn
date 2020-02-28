import numpy as np

class Sigmoid:
    @staticmethod
    def f(x):
        s = 1/(1 + np.exp(-x))
        return s
    
    @staticmethod
    def g(x):
        s = 1/(1 + np.exp(-x))
        return s * (1 - s)
    

class Tanh:
    @staticmethod
    def f(x):
        return np.tanh(x)
        
    @staticmethod
    def g(x):
        return 1 - np.tanh(x)**2


class RELU:   
    @staticmethod
    def f(x):
        xx = x.copy()
        xx[xx<0] = 0
        return xx
    
    @staticmethod
    def g(x):
        xx = x.copy()
        xx[xx<=0] = 0
        xx[xx>0] = 1
        return xx

    
class Softmax:
    @staticmethod
    def f(x):
        s = np.exp(x - np.max(x, axis=0))
        return s/np.sum(s, axis=0)
    
    @staticmethod
    def g(x):
        s = np.exp(x - np.max(x, axis=0))
        s = s / np.sum(s, axis=0)
        I = np.eye(x.shape[0])
        return s.T * (I - s)
    

class Softmax_nograd:
    @staticmethod
    def f(x):
        s = np.exp(x - np.max(x, axis=0))
        return s/np.sum(s, axis=0)
    
    @staticmethod
    def g(x):
        return 1