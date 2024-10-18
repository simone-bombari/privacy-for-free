import numpy as np
import argparse
import time
import os
import json
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from utils_NN_JAX import load_mnist, train, initialize_params

print('Device: ', jax.devices())


parser = argparse.ArgumentParser()
parser.add_argument('--k')
parser.add_argument('--width')
parser.add_argument('--lr')
parser.add_argument('--n')
args = parser.parse_args()

width = int(args.width)
k = int(args.k)
time.sleep(k)

save_dir = os.path.join('NN', 'DP-GD-crossval-grid')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n = int(args.n)
key = jrandom.PRNGKey(k)

train_data, train_labels, val_data, val_labels, test_data, test_labels = load_mnist(n, key)

input_dim = train_data.shape[1]
output_dim = 10

# epsilons = [1, 2, 4]
# eps = epsilons[k % 3]
# k = k // 3

eps = 1

lr = float(args.lr)
batch_size = 500  # to fit in 48 Gb for width = 1000

delta = 1 / n

clip_values = np.logspace(-1, 3, 20)
T_max = 500.01 / lr
Ts = np.logspace(0, np.log10(T_max), 20)
Ts = np.unique(np.round(Ts).astype(int)).tolist()
print_every = 10

for T in Ts:
    for clip_value in clip_values:

        print("------------------------------------")
        print(f"Training with width={width}, lr={lr}, batch_size={batch_size}, eps={eps}, T={T}, clip_value={clip_value}")
        print("------------------------------------")
    
        params = initialize_params(input_dim, width, output_dim, key)
        sigma = jnp.sqrt(T) * jnp.sqrt(8 * jnp.log(1 / delta)) / eps  # notice that we do not have learning_rate here, check training_step
        
        start_time = time.time()
        
        metrics = train(params, lr, train_data, train_labels, val_data, val_labels, test_data, test_labels, T, batch_size, sigma, clip_value, print_every, key)

        end_time = time.time()
        total_time = end_time - start_time
  
        data = {
            'width': width,
            'lr': lr,
            'batch_size': batch_size,
            'eps': eps,
            'clip_value': f"{clip_value:.2f}",
            'T': T,
            'total_time': f"{total_time:.2f}",
            'final_test_accuracy': metrics['test_accuracies'][-1],
            'final_val_accuracy': metrics['val_accuracies'][-1],
            'final_train_loss': metrics['train_losses'][-1]
        }

        with open(os.path.join(save_dir, f'{width}_{lr}_{clip_value:.2f}_{T}_' + args.k + '.json'), 'w') as f:
            json.dump(data, f)
