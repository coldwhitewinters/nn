import numpy as np
import numpy.random as rnd

#%% Arquitectura de la red

class Net:
    def __init__(self, neurons):
        self.neurons = neurons
        self.L = len(neurons)
        self.w, self.b, self.dw, self.db, self.a, self.z = \
        [self.L*[0] for k in range(6)]
        
    def feedforward(self, x):
        self.a[0] = x 
        self.z[0] = x
        for k in range(1, self.L-1):
            self.z[k] = self.w[k] @ self.a[k-1] + self.b[k]
            self.a[k] = Tanh.f(self.z[k])
        self.z[-1] = self.w[-1] @ self.a[-2] + self.b[-1]
        self.a[-1] = Sigmoid.f(self.z[-1])
        return self.a[-1]

    def backprop(self, x, y):
        self.feedforward(x)
        delta = (self.a[-1] - y)
        self.dw[-1] = delta @ self.a[-2].T
        self.db[-1] = np.sum(delta, axis=1, keepdims=True)
        for k in reversed(range(1, self.L-1)):
            delta = (self.w[k+1].T @ delta) * Tanh.g(self.z[k])
            self.dw[k] = delta @ self.a[k-1].T
            self.db[k] = np.sum(delta, axis=1, keepdims=True)


#%% Inicializaciones

def xavier_init(net):    
    for k in range(1, net.L):
        in_size = net.neurons[k-1]
        out_size = net.neurons[k]
        net.w[k] = np.sqrt(2/(in_size+out_size)) * rnd.randn(out_size, in_size)
        net.b[k] = np.zeros((out_size, 1))

def lecun_init(net):
    for k in range(1, net.L):
        in_size = net.neurons[k-1]
        out_size = net.neurons[k]
        net.w[k] = np.sqrt(1/in_size) * rnd.randn(out_size, in_size)
        net.b[k] = np.zeros((out_size, 1))
        
def he_init(net):
    for k in range(1, net.L):
        in_size = net.neurons[k-1]
        out_size = net.neurons[k]
        net.w[k] = np.sqrt(2/in_size) * rnd.randn(out_size, in_size)
        net.b[k] = 0.1*np.ones((out_size, 1))
        
def normal_init(net):
    for k in range(1, net.L):
        in_size = net.neurons[k-1]
        out_size = net.neurons[k]
        net.w[k] = rnd.randn(out_size, in_size)
        net.b[k] = rnd.randn(out_size, 1)
        
def zero_init(net):
    for k in range(1, net.L):
        in_size = neurons[k-1]
        out_size = neurons[k]
        net.w[k] = np.zeros((out_size, in_size))
        net.b[k] = np.zeros((out_size, 1))
        
        
#%% Optimizadores
        
def SGD(net, trng_set, val_set, loss_fn, 
        epochs=30, bat_size=100, rate=1, weight_decay=0):
    x, y = trng_set
    trng_size = x.shape[1]
    for n in range(epochs):
        shf_idx = rnd.permutation(trng_size)
        x, y = x[:, shf_idx], y[:, shf_idx]
        for k in range(0, trng_size, bat_size):
            net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for k in range(1, net.L):
                decay = (1 - rate*weight_decay/trng_size)
                net.w[k] =  decay * net.w[k] - rate/bat_size * net.dw[k]
                net.b[k] =  net.b[k] - rate/bat_size * net.db[k]
        acc, loss = performance(net, val_set, loss_fn)
        print('epoch: %2d | acc: %2.2f%% | loss: %.5f' % (n+1, acc*100, loss))
        
def SGDMomentum(net, trng_set, val_set, loss_fn,
                epochs=30, bat_size=100, rate=0.001, momentum=0.9):
    x, y = trng_set
    trng_size = x.shape[1]
    vw, vb = [0], [0]
    for k in range(1, net.L):
        vw.append(np.zeros(net.w[k].shape))
        vb.append(np.zeros(net.b[k].shape))
    for n in range(epochs):
        shf_idx = np.random.permutation(trng_size)
        x, y = x[:, shf_idx], y[:, shf_idx]
        for k in range(0, trng_size, bat_size):
            net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for k in range(1, net.L):
                vw[k] = momentum * vw[k] - rate/bat_size * net.dw[k]
                vb[k] = momentum * vb[k] - rate/bat_size * net.db[k]
                net.w[k] =  net.w[k] + vw[k]
                net.b[k] =  net.b[k] + vb[k]
        acc, loss = performance(net, val_set, loss_fn)
        print('epoch: %2d | acc: %2.2f%% | loss: %.5f' % (n+1, acc*100, loss))
        

#%% Monitoreo del rendimiento
        
def performance(net, test_set, loss_fn):
    x, y = test_set
    test_size = x.shape[1] 
    net.feedforward(x)
    pred = np.argmax(net.a[-1], axis=0)
    label = np.argmax(y, axis=0)
    acc = np.mean(pred == label)
    loss = loss_fn(net.a[-1], y) / test_size
    return acc, loss


#%% Costos
    
def quad_cost(a, y):
    return 1/2 * np.sum((a - y)**2)

def cross_entropy_cost(a, y):
    return -np.sum(np.nan_to_num(y*np.log(a) + (1 - y)*np.log(1 - a)))

def nll_cost(a, y):
    return -np.sum(y * np.log(a))


#%% Activaciones
    
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
    
class Relu:   
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


#%% Carga de datos
    
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

#%% Main
    
trng_set, val_set = load_data()
neurons = [784, 400, 100, 10]
net = Net(neurons)
xavier_init(net)
SGD(net, trng_set, val_set, cross_entropy_cost, 
    epochs=30, bat_size=32, rate=0.1, weight_decay=3)

        