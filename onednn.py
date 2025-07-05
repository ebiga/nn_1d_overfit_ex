import os
import silence_tensorflow.auto
import matplotlib
matplotlib.use('TkAgg')
matplotlib.set_loglevel('critical')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from keras import layers, regularizers
from keras.utils import get_custom_objects
from keras.callbacks import ReduceLROnPlateau

# set floats and randoms
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)



## FUNCTION: Compute Laplacians for 1D, cell centred, for normal or staggered meshes
#_ The Laplacians are computed along the diagonals, so we can have only one staggered mesh
#_ It's never enough any effort to make life easier...
def compute_Laplacian(f_orig, f_stag):

    if f_orig is not f_stag:
        # If the arrays are not the same, we have a staggered mesh with the original mesh at the centre/corners
        #_ We compute the Laplacian with a 5-point stencil
        grid_spacing = 0.5

        delta = grid_spacing**2.

        dsf_dD1s = np.abs(f_orig[2:] + f_orig[ :-2] - f_stag[1:] - f_stag[ :-1]) \
                       / (f_orig[2:] + f_orig[ :-2] + f_stag[1:] + f_stag[ :-1]) / (3.*delta)

        return dsf_dD1s

    else:
        # This is the original training mesh processing, fstag = f_orig
        #_ We apply a 3-point stencil to compute the Laplacian
        grid_spacing = 1.

        delta = grid_spacing**2.

        dsf_dD1s = np.abs(f_orig[2:] + f_orig[ :-2] - 2. * f_orig[1:-1]) \
                       / (f_orig[2:] + f_orig[ :-2] + 2. * f_orig[1:-1]) / delta

        return dsf_dD1s


## FUNCTION: Train a model
def minimise_NN_RMSE(model, DATAX, DATAF):
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=lr))
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=500, cooldown=100, verbose=1, min_lr=1e-6)
    history = model.fit(
        DATAX,
        DATAF,
        verbose=0, epochs=epocs, batch_size=batchs,
        callbacks=[reduce_lr],
        )
    loss = np.log(history.history['loss'])
    return model, loss


## FUNCTION: Build a model
def get_me_a_model(nn_layers, input_shape, activfunk):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for nn in nn_layers:
        x = layers.Dense(nn, activation=activfunk, kernel_initializer='he_normal')(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


## FUNCTION: Compute the rms of the mean
def check_mean(mean, refd):
    delta = refd - mean

    rms_check = np.sqrt( np.mean( delta**2. ) )
    mae_check = np.mean( np.abs(delta) )
    max_check = np.max( np.abs(delta) )

    msg = "Errors: rms, mean, max: " + f"\t{rms_check:.3e};\t {mae_check:.3e};\t {max_check:.3e}\n"
    return msg


## FUNCTION: Custom activation function, this is like a leaky ELU
class LeakyELU(tf.keras.layers.Layer):
    def __init__(self, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        pos = tf.where(inputs > 0, inputs, tf.zeros_like(inputs))
        neg = tf.where(
            inputs <= 0,
            tf.exp((1.0 - self.beta) * inputs) - 1.0 + self.beta * inputs,
            tf.zeros_like(inputs)
        )
        return pos + neg

    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config


## FUNCTION: Our function
def my_func(x, name='tanh'):

    # Evaluate one of several benchmark functions at given input points x.

    # Parameters:
    # - name: str, one of ['runge', 'gaussian_bump', 'step_plus_slope', 'piecewise_sine', 'sparse_poly']
    # - x: np.ndarray, shape (N,), input values where function will be evaluated

    # Returns:
    # - y: np.ndarray, shape (N,), function values at input x

    if name == 'runge':
        # f(x) = 1 / (1 + 25x^2), x ∈ [-1, 1], recommended N: 11–50
        y = 1 / (1 + 25 * x**2)

    elif name == 'gaussian_bump':
        # f(x) = sin(2πx) + 0.2 * exp(-100(x - 0.5)^2), x ∈ [0, 1], recommended N: 50–200
        y = np.sin(2 * np.pi * x) + 0.2 * np.exp(-100 * (x - 0.5)**2)

    elif name == 'step_plus_slope':
        # f(x) = sigmoid(10(x - 0.5)) + x, x ∈ [0, 1], recommended N: 50–200
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + x

    elif name == 'piecewise_sine':
        # f(x) = sin(2πx) for x < 0.5, sin(2πx) + 0.5 otherwise, x ∈ [0, 1], recommended N: 50–200
        y = np.where(x < 0.5, np.sin(2 * np.pi * x), np.sin(2 * np.pi * x) + 0.5)

    elif name == 'sparse_poly':
        # f(x) = x^3 - 2x^2 + x, x ∈ [0, 1], recommended N: 10–30
        y = x**3 - 2 * x**2 + x
    
    elif name == 'tanh':
        # 7 and 14 points, x ∈ [0, 1]
        y = 0.5 + 0.5*np.tanh(5.0*x)
    
    elif name == 'pointonex':
        # 12 points, x ∈ [0, 1]
        y = np.power(0.1, x)

    return y

        




## DEFINITIONS
#_ function
x_min = -1.0
x_max =  1.0
n_pts =  14

#_ network
lr = 0.001
epocs = 15000
batchs = 3
nn_layers = [240] * 8

#_ general shiz
num_layers = len(nn_layers)
layer_size = nn_layers[0]
layer_tag  = f"{num_layers}x{layer_size}"

actifun = {
    'elu': tf.keras.activations.elu,
    'silu': tf.keras.activations.silu,
    'relu': tf.keras.activations.relu,
    'leaky_relu': tf.keras.activations.leaky_relu,
    'lelu-0.3': LeakyELU(beta=0.3),
    'lelu-0.4': LeakyELU(),
    'lelu-0.6': LeakyELU(beta=0.6),
    'softplus': tf.keras.activations.softplus,
    'tanh': tf.keras.activations.tanh,
}


# Build the dataset space
DATAX = np.linspace(x_min, x_max, n_pts)
DATAF = my_func(DATAX)

reffile = pd.DataFrame(DATAX)
reffile['f'] = DATAF
reffile.to_csv('ref_data.csv', index=False)

# staggered and laplacian
STAGX = ( DATAX[:-1] + DATAX[1:] ) / 2.

laplacian_dataf = compute_Laplacian(DATAF, DATAF)


# Loop for several activation functions
for an, af in actifun.items():
    print(f"Doing {an}")
    dafolda = os.path.join(layer_tag, an)
    os.makedirs(dafolda, exist_ok=True)

    flightlog = open(os.path.join(dafolda, 'log.txt'), 'w')

    # build and train the model
    gisele = get_me_a_model(nn_layers, [1], af)
    gisele, loss = minimise_NN_RMSE(gisele, DATAX, DATAF)


    # how bout data
    ymean = gisele.predict(DATAX).reshape(-1)
    msg = check_mean(ymean, DATAF)
    print(msg)
    flightlog.write(msg+'\n')

    ystag = gisele.predict(STAGX).reshape(-1)
    laplacian_stagf = compute_Laplacian(ymean, ystag)

    loss_m = np.sqrt(np.mean((laplacian_stagf - laplacian_dataf)**2.))
    msg = f"RMSE of the Laplacians: {loss_m:.3e}"
    print(msg)
    flightlog.write(msg+'\n')


    # print da loss
    plt.plot(np.array(loss), label=f"Training Loss (RMSE of the Laplacians: {loss_m:.3e})")
    plt.xlabel('Epochs')
    plt.ylabel('Log(Loss)')
    plt.title('Loss Convergence')
    plt.legend()
    plt.savefig(os.path.join(dafolda, 'convergence.png'), format='png', dpi=1200)
    plt.close()


    # print da data
    xplot = np.linspace(x_min, x_max, 999)
    yplot = my_func(xplot)
    ypred = gisele.predict(xplot).reshape(-1)

    predfile = pd.DataFrame(xplot)
    predfile['f'] = yplot
    predfile['f_pred'] = ypred
    predfile.to_csv(os.path.join(dafolda, 'test_data_out.csv'), index=False)

    plt.scatter(DATAX, DATAF, label='training')
    plt.plot(xplot, yplot, label='ref')
    plt.plot(xplot, ypred, label='pred')
    plt.legend()
    plt.savefig(os.path.join(dafolda, 'result.png'), format='png', dpi=1200)
    plt.close()
