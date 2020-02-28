import numpy as np
import numpy.random as rnd

#%% Arquitectura de la red

neurons = [784, 400, 100, 10]
L = len(neurons)
w, b, dw, db, a, z = [L*[0] for k in range(6)]

def feedforward(x):
    a[0] = x 
    z[0] = x
    for k in range(1, L-1):
        z[k] = w[k] @ a[k-1] + b[k]
        a[k] = np.tanh(z[k])
    z[-1] = w[-1] @ a[-2] + b[-1]
    a[-1] = sigmoid(z[-1])
    return a[-1]

def backprop(x, y):
    feedforward(x)
    delta = (a[-1] - y)
    dw[-1] = delta @ a[-2].T
    db[-1] = np.sum(delta, axis=1, keepdims=True)
    for k in reversed(range(1, L-1)):
        delta = (w[k+1].T @ delta) * tanh_prime(z[k])
        dw[k] = delta @ a[k-1].T
        db[k] = np.sum(delta, axis=1, keepdims=True)

#%% Inicializaciones
        
def xavier_init():    
    for k in range(1, L):
        in_size = neurons[k-1]
        out_size = neurons[k]
        w[k] = np.sqrt(2/(in_size+out_size)) * rnd.randn(out_size, in_size)
        b[k] = np.zeros((out_size, 1))  

def lecun_init():
    for k in range(1, L):
        in_size = neurons[k-1]
        out_size = neurons[k]
        w[k] = np.sqrt(1/in_size) * rnd.randn(out_size, in_size)
        b[k] = np.zeros((out_size, 1))
        
def normal_init():
    for k in range(1, L):
        in_size = neurons[k-1]
        out_size = neurons[k]
        w[k] = rnd.randn(out_size, in_size)
        b[k] = rnd.randn(out_size, 1)
        
def zero_init():
    for k in range(1, L):
        in_size = neurons[k-1]
        out_size = neurons[k]
        w[k] = np.zeros((out_size, in_size))
        b[k] = np.zeros((out_size, 1))
 

#%% Método de entrenamiento
        
def SGD(trng_set, val_set, loss_fn, epochs=30, bat_size=100, eta=1):
    x, y = trng_set
    trng_size = x.shape[1]
    for n in range(epochs):
        shf_idx = rnd.permutation(trng_size)
        x, y = x[:, shf_idx], y[:, shf_idx]
        for k in range(0, trng_size, bat_size):
            backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for k in range(1, L):
                w[k] =  w[k] - eta/bat_size * dw[k]
                b[k] =  b[k] - eta/bat_size * db[k]
        acc, loss = performance(val_set, loss_fn)
        print('epoch: %2d | acc: %2.2f%% | loss: %.5f' % (n+1, acc*100, loss))

#%% Monitoreo del rendimiento
        
def performance(test_set, loss_fn):
    x, y = test_set
    test_size = x.shape[1] 
    feedforward(x)
    pred = np.argmax(a[-1], axis=0)
    label = np.argmax(y, axis=0)
    acc = np.mean(pred == label)
    loss = loss_fn(a[-1], y) / test_size
    return acc, loss

#%% Costo
    
def quad_cost(a, y):
    return 1/2 * np.sum((a - y)**2)

def quad_grad(a, y):
    return a - y

def cross_entropy_cost(a, y):
    return -np.sum(y*np.log(a) + (1 - y)*np.log(1 - a))

#%% Activaciones y derivadas
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

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
xavier_init()
SGD(trng_set, val_set, cross_entropy_cost, epochs=30, bat_size=32, eta=0.1)

        