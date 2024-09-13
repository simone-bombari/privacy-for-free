import numpy as np
import argparse
import time
import os
import json

import jax.numpy as jnp
import jax.random as jrandom
import jax

from utils_NN_JAX import *

print('Device: ', jax.devices())


parser = argparse.ArgumentParser()
parser.add_argument('--k')
parser.add_argument('--clipping_mode')
args = parser.parse_args()

clipping_mode = args.clipping_mode
k = int(args.k)
time.sleep(k)

save_dir = os.path.join('NN', '13_09_grid')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_data, train_labels, test_data, test_labels = load_mnist()

input_dim = train_data.shape[1]
output_dim = 10
key = jrandom.PRNGKey(0)

widths = [40, 100, 200, 400, 1000, 2000]
width = widths[k % 6]
k = k // 6

lr = 1
batch_size = 800000 // width  # to fit in 64 Gb I suppose  # maybe try 120 * 10^4 over 96Gb of memory.

n = 60000
delta = 1 / n

epsilons = [1, 4]
eps = epsilons[k % 2]
k = k // 2

if clipping_mode == 'layer':
    clip_values_1 = np.logspace(0, 1, 5)
    # clip_value_1 = clip_values_1[k % 5]
    # k = k // 5

    clip_values_2 = np.logspace(0, 1.5, 5)
    # clip_value_2 = clip_values_2[k % 5]
    # k = k // 5
elif clipping_mode == 'global':
    clip_values_1 = np.logspace(0, 2, 20)
    clip_values_2 = [None]

Ts = np.logspace(1, 3, 10)
print_every = 10

for T in Ts:
    T = int(T)
    
    for clip_value_1 in clip_values_1:
        for clip_value_2 in clip_values_2:

            clip_values = (clip_value_1, clip_value_2)
            
            print("------------------------------------")
            print(f"Clipping mode: {clipping_mode}")
            print("------------------------------------")
            print(f"Training with width={width}, lr={lr}, batch_size={batch_size}, eps={eps}, T={T}, clip_value_1={clip_value_1}, clip_value_2={clip_value_2}")
            print("------------------------------------")
        
            params = initialize_params(input_dim, width, output_dim, key)
        
            sigma = jnp.sqrt(lr * T) * jnp.sqrt(8 * jnp.log(1 / delta)) / eps
            
            if clipping_mode == 'layer':  # I am effectively doing 2T updates
                sigma = sigma * jnp.sqrt(2)
            
            metrics = train(params, lr, train_data, train_labels, test_data, test_labels,
                      T, batch_size, sigma, clip_values, print_every, clipping_mode=clipping_mode, clip_debug=False)
        
            data = {
                'width': width,
                'lr': lr,
                'batch_size': batch_size,
                'eps': eps,
                'clip_value_1': clip_value_1,
                'clip_value_2': clip_value_2,
                'T': T,
                'test_accuracies': metrics['test_acc'],
                'train_losses': metrics['train_loss'],
                'clipping_mode': clipping_mode
            }
        
            with open(os.path.join(save_dir, f'{clipping_mode}_{width}_{clip_value_1}_{clip_value_2}_{T}_' + args.k + '.json'), 'w') as f:
                json.dump(data, f)
