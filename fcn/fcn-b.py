import numpy as np

#%% Arquitectura de la red

class Net:
    def __init__(self, neurons, act=None, out=None):
        self.neurons = neurons
        self.L = len(neurons)
        self.act = act
        self.out = out
        self.w, self.dw,\
        self.b, self.db,\
        self.a, self.z = [(self.L)*[0] for k in range(6)]

    def feedforward(self, x):
        self.a[0] = x
        self.z[0] = x
        for k in range(1, self.L-1):
            self.z[k] = self.w[k] @ self.a[k-1] + self.b[k]
            self.a[k] = self.act.f(self.z[k])
        self.z[-1] = self.w[-1] @ self.a[-2] + self.b[-1]
        self.a[-1] = self.out.f(self.z[-1])
        return self.a[-1]

    def backprop(self, x, y):
        self.feedforward(x)
        delta = (self.a[-1] - y)
        self.dw[-1] = delta @ self.a[-2].T
        self.db[-1] = np.sum(delta, axis=1, keepdims=True)
        for k in reversed(range(1, self.L-1)):
            delta = (self.w[k+1].T @ delta) * self.act.g(self.z[k])
            self.dw[k] = delta @ self.a[k-1].T
            self.db[k] = np.sum(delta, axis=1, keepdims=True)

#%% Inicializacion

def init_weights(net, mode='lecun'):
    fan_in = lambda k: net.neurons[k-1]
    fan_out = lambda k: net.neurons[k]
    if mode == 'lecun':
        factor = lambda k: np.sqrt(1/fan_in(k))
    elif mode == 'xavier':
        factor = lambda k: np.sqrt(2/(fan_in(k) + fan_out(k)))
    elif mode == 'he':
        factor = lambda k: np.sqrt(2/fan_in(k))
    elif mode == 'normal':
        factor = lambda k: 1
    elif mode == 'zero':
        factor = lambda k: 0
    else:
        raise Exception('Wrong mode')
    for k in range(1, net.L):
        net.w[k] = factor(k) * np.random.randn(fan_out(k), fan_in(k))
        net.b[k] = 0.1*np.ones((fan_out(k), 1))
        #net.b[k] = np.zeros((fan_out(k), 1))

#%% Optimizadores

def SGD(net, trng_set, val_set, loss_fn,
        epochs=30, bat_size=100, rate=0.001, weight_decay=0):
    x, y = trng_set
    trng_size = x.shape[1]
    for n in range(epochs):
        shf_idx = np.random.permutation(trng_size)
        x, y = x[:, shf_idx], y[:, shf_idx]
        for k in range(0, trng_size, bat_size):
            net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for k in range(1, net.L):
                decay_factor = (1 - rate*weight_decay/trng_size)
                net.w[k] =  decay_factor * net.w[k] - rate/bat_size * net.dw[k]
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

def RMSProp(net, trng_set, val_set, loss_fn,
            epochs=30, bat_size=100, rate=0.001, rho=0.9):
    x, y = trng_set
    trng_size = x.shape[1]
    rw, rb = [0], [0]
    for k in range(1, net.L):
        rw.append(np.zeros(net.w[k].shape))
        rb.append(np.zeros(net.b[k].shape))
    for n in range(epochs):
        shf_idx = np.random.permutation(trng_size)
        x, y = x[:, shf_idx], y[:, shf_idx]
        for k in range(0, trng_size, bat_size):
            net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for k in range(1, net.L):
                gw, gb = net.dw[k]/bat_size, net.db[k]/bat_size
                rw[k] = rho * rw[k] + (1 - rho) * gw**2
                rb[k] = rho * rb[k] + (1 - rho) * gb**2
                net.w[k] =  net.w[k] - rate/(np.sqrt(rw[k]) + 1e-7) * gw
                net.b[k] =  net.b[k] - rate/(np.sqrt(rb[k]) + 1e-7) * gb
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

#%% Costos

def quad_cost(a, y):
    return 1/2 * np.sum((a - y)**2)

def cross_entropy_cost(a, y):
    return -np.sum(np.nan_to_num(y*np.log(a) + (1 - y)*np.log(1 - a)))

#%% Main

trng_set, val_set = load_data()
neurons = [784, 400, 100, 10]
net = Net(neurons, act=Tanh, out=Sigmoid)
init_weights(net, mode='xavier')
SGD(net, trng_set, val_set, cross_entropy_cost, rate=0.1)
#SGDMomentum(net, trng_set, val_set, cross_entropy_cost, rate=0.1, momentum=0.5)
#RMSProp(net, trng_set, val_set, cross_entropy_cost, rate=0.001, rho=0.9)

