import numpy as np
import argparse
import time
import os
import json
import sys

import jax.numpy as jnp
from jax import grad, jit, vmap, nn
import optax
import jax.random as jrandom
import jax
from jax.tree_util import tree_map, tree_leaves

from NN_JAX_utils import *

print('Device: ', jax.devices())


parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)
time.sleep(k)


save_dir = os.path.join('NN', '___')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_data, train_labels, test_data, test_labels = load_mnist()
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

input_dim = train_data.shape[1]
output_dim = 10
key = jrandom.PRNGKey(0)

# Clr = 1
# lr = 1000 * Clr / width

training_modes = ['full', 'no_noise', 'no_clip', 'no_clip_with_noise']

batch_sizes = [1000, 500]
batch_size = batch_sizes[k % 2]
k = k // 2

lrs = [1, 0.1]
lr = lrs[k % 2]
k = k // 2

# widths = [100, 1000, 10000]
# width = widths[k % 3]
# k = k // 3

width = 100

n = 60000
delta = 1 / n

eps = 1
(clip_value_1, clip_value_2) = (0.3, 0.3)

# epsilons = [1, 4]
# eps = epsilons[k % 2]
# k = k // 2

# clip_values_1 = np.logspace(-1, 2, 4)
# clip_value_1 = clip_values_1[k % 4]  # 4 options
# k = k // 4

# clip_values_2 = np.logspace(0, 3, 4)
# clip_value_2 = clip_values_2[k % 4]  # 4 options
# k = k // 4

clip_values = (clip_value_1, clip_value_2)

Ts = np.logspace(1, 3, 3)

print_every = 10

for T in Ts:
    T = int(T)
    for training_mode in training_modes:

        print("------------------------------------")
        print(f"{training_mode}")
        print("------------------------------------")
        print(f"Training with width={width}, lr={lr}, batch_size={batch_size}, eps={eps}, T={T}, clip_value_1={clip_value_1}, clip_value_2={clip_value_2}")
        print("------------------------------------")
    
        params = initialize_params(input_dim, width, output_dim, key)
    
        sigma = jnp.sqrt(2) * jnp.sqrt(lr * T) * jnp.sqrt(8 * jnp.log(1 / delta)) / eps
        noise_multiplier = jnp.sqrt(lr) * (2 * 1 / n) * sigma  # The 1 is the clipping constant inside the function 
    
        _, metrics = train(params, lr, train_data, train_labels, test_data, test_labels,
                  T, batch_size, noise_multiplier, clip_values, print_every, training_mode=training_mode)
    
        # print(metrics)
    
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
            'grad_1_norms': metrics['grad_1_norms'],
            'grad_2_norms': metrics['grad_2_norms'],
        }
    
        with open(os.path.join(save_dir, f'{training_mode}_T={T}_' + args.k + '.json'), 'w') as f:
            json.dump(data, f)
