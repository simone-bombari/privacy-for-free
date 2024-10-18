import numpy as np
import time
import os
import json
import time
import argparse

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, nn, debug
import optax
import jax.random as jrandom
from jax.tree_util import tree_map, tree_leaves
from torchvision.datasets import MNIST
from jax.flatten_util import ravel_pytree

from utils_NN_JAX import load_mnist, initialize_params, predict, loss, evaluate

print('Device: ', jax.devices())


@jit
def training_step(params, batched_data, batched_labels, learning_rate):
    
    def single_sample_loss(params, x, y):  # the loss defined in utils will be JIT compiled when calling it over multiple samples
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)
    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))
    
    def process_batch(carry, batch_data_and_labels):
        batch_data, batch_labels = batch_data_and_labels
        batch_grads = batch_grad_fn(params, batch_data, batch_labels)
        batch_sum_grads = tree_map(lambda g: jnp.sum(g, axis=0), batch_grads)
        new_carry = tree_map(lambda acc, batch: acc + batch, carry, batch_sum_grads)
        return new_carry, None

    init_carry = tree_map(lambda p: jnp.zeros_like(p), params)  # initialize the gradients to 0, it will aggregate inside carry looping over the batches

    accumulated_grads, _ = jax.lax.scan(
        process_batch, init_carry, (batched_data, batched_labels)  # Even if I am passing a tuple, scan will loop over the individual arrays
    )

    num_samples = batched_data.shape[0] * batched_data.shape[1]  # Total number of samples, works because batch_size divides n
    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)
    new_params = tree_map(lambda p, g: p - learning_rate * g, params, avg_grads)
    
    return new_params


def train(params, learning_rate, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, printing_at, scheduler):
    
    metrics = {'train_losses': {epoch: None for epoch in printing_at}, 'test_accuracies': {epoch: None for epoch in printing_at}}

    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size
    batched_data = train_data[:num_batches * batch_size].reshape((num_batches, batch_size, -1))
    batched_labels = train_labels[:num_batches * batch_size].reshape((num_batches, batch_size))

    for epoch in range(1, num_epochs + 1):
        if epoch in scheduler.keys():
            learning_rate = scheduler[epoch]
            
        params = training_step(params, batched_data, batched_labels, learning_rate)
        
        if epoch in metrics['train_losses'].keys():
            train_loss = jnp.mean(loss(params, train_data, train_labels))
            test_acc = evaluate(params, test_data, test_labels)
            print(f"Epoch {epoch}: learning rate = {learning_rate}, Train loss = {train_loss}, Test accuracy = {test_acc}", flush=True)
            metrics['train_losses'][epoch] = train_loss.item()
            metrics['test_accuracies'][epoch] = test_acc.item()
            
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument('--k')
parser.add_argument('--width')
parser.add_argument('--lr')
args = parser.parse_args()

save_dir = os.path.join('NN', 'GD-crossval-ns')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

k = int(args.k)

width = int(args.width)
lr = float(args.lr)
ns = [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]

input_dim = 784
output_dim = 10
T = int(1e5)
printing_at = np.unique(np.logspace(0, 5, 1000).astype(int))
printing_at = [int(key) for key in printing_at] # would otherwise give issues in json dumping

key = jrandom.PRNGKey(k)  # only randomness is in the model initialization

scheduler = {}

for n in ns:

    batch_size = min(500000 // width, n)  # to fit in 40 Gb
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_mnist(n, key)  # n training samples and 60000 - n val samples
    
    print(f'-------------------------------------------- Training with width = {width} --------------------------------------------')
    
    params = initialize_params(input_dim, width, output_dim, key)
    start_time = time.time()
        
    metrics = train(params, lr, train_data, train_labels, test_data, test_labels,
              T, batch_size, printing_at, scheduler)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'--------------------------- End of training. Total time = {total_time:.2f} ---------------------------')
    
    metrics['lr'] = lr
    metrics['width'] = width
    metrics['n'] = n
    metrics['total_time'] = total_time
    
    print(metrics)
    
    with open(os.path.join(save_dir, f'GD_n={n}_width={width}_lr={lr}_' + str(k) + '.json'), 'w') as f:
        json.dump(metrics, f)
