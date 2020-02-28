import numpy as np

def SGD(net, trng_set, val_set, epochs, bat_size, eta, lmbda):
    x, y = trng_set
    trng_size = x.shape[1]
    for n in range(epochs):       
        shuffle_data(x, y)
        for k in range(0, trng_size, bat_size):
            net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
            for layer in net.hidden:
                decay = (1 - eta*lmbda/trng_size)
                layer.w = decay * layer.w - eta/bat_size * layer.dw
                layer.b = layer.b - eta/bat_size * layer.db
        acc, loss = net.performance(val_set)
        print('epoch: %2d | loss: %.3f | acc: %.3f' % (n+1, loss, acc))
        
def SGDMomentum(net, trng_set, val_set, epochs, bat_size, eta, rho, lmbda):
        x, y = trng_set
        trng_size = x.shape[1]
        for n in range(epochs):
            shuffle_data(x, y)
            for layer in net.layers:
                layer.vw = np.zeros(layer.w.shape)
                layer.vb = np.zeros(layer.b.shape)
            for k in range(0, trng_size, bat_size):
                net.backprop(x[:, k:k+bat_size], y[:, k:k+bat_size])
                for layer in net.hidden:
                    decay = (1 - eta*lmbda/trng_size)
                    layer.vw = rho*layer.vw + layer.dw / bat_size
                    layer.vb = rho*layer.vb + layer.db / bat_size
                    layer.w = decay * layer.w - eta * layer.vw
                    layer.b = layer.b - eta * layer.vb
            acc, loss = net.performance(val_set)
            print('epoch: %2d | loss: %.3f | acc: %.3f' % (n+1, loss, acc))
            
def shuffle_data(x, y):
    shf_idx = np.random.permutation(x.shape[1])
    x = x[:, shf_idx]
    y = y[:, shf_idx]
