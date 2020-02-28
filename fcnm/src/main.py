import numpy as np
import modules
from modules import FC
import optimizers as optim
import activations as act

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.hidden = layers[:-1]
        self.output = layers[-1]
        
    def feedforward(self, x):
        a = x
        for layer in self.layers:
            a = layer.feedforward(a) 
        return a
    
    def backprop(self, x, y):
        a = self.feedforward(x)
        upstream = self.output.backprop(a, y)
        for layer in reversed(self.hidden):
            upstream = layer.backprop(upstream)
              
    def performance(self, test_set):
        x, y = test_set
        test_size = x.shape[1] 
        a = self.feedforward(x)
        pred = np.argmax(a, axis=0)
        label = np.argmax(y, axis=0)
        acc = np.mean(pred == label)
        loss = self.output.cost(a, y) / test_size
        return acc, loss
    
def load_data(trng_size=50000, val_size=10000, 
              shuffle=False, normalize=True):
    total_size = trng_size + val_size
    x, y = np.load('../data/trng_data.npy')
    if shuffle:
        shf_idx = np.random.permutation(x.shape[1])
        x, y = x[:, shf_idx], y[:, shf_idx]
    if normalize:
        x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
    trng_set = (x[:,0:trng_size], y[:,0:trng_size])
    val_set = (x[:,trng_size:total_size], y[:,trng_size:total_size])
    return trng_set, val_set
    
### MAIN ###

trng_set, val_set = load_data()
 
layers = [FC(784, 1000, act.RELU), 
          FC(1000, 500, act.RELU),
          FC(500, 100, act.RELU),
          modules.NLLSoftmax(100, 10)]

net = Network(layers)

optim.SGDMomentum(net, trng_set, val_set, 30, 100, 0.1, 0.9, 1)


        