import jax.numpy as jnp
from jax import grad, jit, vmap, nn, debug
import optax
import jax.random as jrandom
from jax.tree_util import tree_map, tree_leaves
from torchvision.datasets import MNIST
import numpy as np


def load_mnist(data_dir='./data'):
    mnist_train = MNIST(root=data_dir, train=True, download=False)
    mnist_test = MNIST(root=data_dir, train=False, download=False)

    X_train = np.array([np.array(image) for image in mnist_train.data], dtype=np.float32)
    y_train = np.array(mnist_train.targets, dtype=int)
    X_test = np.array([np.array(image) for image in mnist_test.data], dtype=np.float32)
    y_test = np.array(mnist_test.targets, dtype=int)

    X_train /= 255.0
    X_test /= 255.0

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    train_data = jnp.array(X_train)
    test_data = jnp.array(X_test)
    train_labels = jnp.array(y_train)
    test_labels = jnp.array(y_test)

    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)
    train_data = (train_data - mean) / (std + 1e-8)  # I manually implement normalization, standard in Pytorch dataloading
    test_data = (test_data - mean) / (std + 1e-8)

    return train_data, train_labels, test_data, test_labels


# ----------------------------------------------------------------------


def initialize_params(input_dim, hidden_dim, output_dim, key):
    key1, key2 = jrandom.split(key)
    stddev1 = jnp.sqrt(2.0 / input_dim)
    stddev2 = jnp.sqrt(2.0 / hidden_dim)
    V_1 = jrandom.normal(key1, (hidden_dim, input_dim)) * stddev1
    V_2 = jrandom.normal(key2, (output_dim, hidden_dim)) * stddev2
    return (V_1, V_2)

initialize_params = jit(initialize_params, static_argnums=(0, 1, 2))  # Probably should keep width fixed in a single script call


@jit
def predict(params, x):
    V_1, V_2 = params
    pre_act = jnp.matmul(x, V_1.T)
    act = nn.relu(pre_act)
    out = jnp.matmul(act, V_2.T)
    return out


@jit
def accumulate_grads(accumulated_grads, batch_grads):
    return tree_map(lambda acc, batch: acc + batch, accumulated_grads, batch_grads)


# ----------------------------------------------------------------------


def compute_batch_clipped_grads(params, x, y, clip_norms, clip_debug=False):
    def single_sample_loss(params, x, y):
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)

    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))
    per_sample_grads = batch_grad_fn(params, x, y)

    if clip_debug:
        for idx, (g, c) in enumerate(zip(tree_leaves(per_sample_grads), clip_norms)):
            norms = jnp.linalg.norm(g.reshape(g.shape[0], -1), axis=1)
            avg_norm = jnp.mean(norms)
            debug.print("Parameter {idx} before clipping: Average per-sample gradient norm = {avg_norm}", idx=idx, avg_norm=avg_norm)

    def clip_gradient(g, clip_norm):
        grad_norm = jnp.linalg.norm(g)
        return jnp.where(grad_norm <= clip_norm, g, g * (clip_norm / grad_norm))

    per_sample_clipped_grads = tree_map(
        lambda g, c: vmap(clip_gradient, in_axes=(0, None))(g, c),
        per_sample_grads, clip_norms
    )

    if clip_debug:
        for idx, (g, c) in enumerate(zip(tree_leaves(per_sample_clipped_grads), clip_norms)):
            norms = jnp.linalg.norm(g.reshape(g.shape[0], -1), axis=1)
            avg_norm = jnp.mean(norms)
            debug.print("Parameter {idx} after clipping: Average per-sample gradient norm = {avg_norm}", idx=idx, avg_norm=avg_norm)

    batch_sum_clipped_grads = tree_map(lambda g: jnp.sum(g, axis=0), per_sample_clipped_grads)
    return batch_sum_clipped_grads

compute_batch_clipped_grads = jit(compute_batch_clipped_grads, static_argnums=(4))


def training_step_full(params, train_data, train_labels, learning_rate, sigma, clip_norms, key, batch_size, clip_debug=False):
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size
    accumulated_grads = tree_map(lambda p: jnp.zeros_like(p), params)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        batch_grads = compute_batch_clipped_grads(params, batch_data, batch_labels, clip_norms, clip_debug)
        accumulated_grads = accumulate_grads(accumulated_grads, batch_grads)

    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)
    keys = tuple(jrandom.split(key, len(tree_leaves(avg_grads))))
    noises = tree_map(
        lambda c, k, g: jnp.sqrt(learning_rate) * (2 * c / num_samples) * sigma * jrandom.normal(k, g.shape),
        clip_norms, keys, avg_grads
    )
    new_params = tree_map(lambda p, g, n: p - learning_rate * g + n, params, avg_grads, noises)
    return new_params

training_step_full = jit(training_step_full, static_argnums=(7, 8))


# ----------------------------------------------------------------------


def compute_batch_global_clipped_grads(params, x, y, clip_norm, clip_debug=False):
    def single_sample_loss(params, x, y):
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)

    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))
    per_sample_grads = batch_grad_fn(params, x, y)

    def squared_l2_norm(g):
        return jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim)))

    per_sample_squared_norms = tree_map(squared_l2_norm, per_sample_grads)
    per_sample_total_squared_norms = sum(tree_leaves(per_sample_squared_norms))
    per_sample_grad_norms = jnp.sqrt(per_sample_total_squared_norms)

    if clip_debug:
        avg_grad_norm_before = jnp.mean(per_sample_grad_norms)
        debug.print("Average per-sample gradient norm before clipping (global): {x}", x=avg_grad_norm_before)

    clipping_factors = jnp.minimum(1.0, clip_norm / per_sample_grad_norms)

    def clip_gradient(g):
        broadcast_shape = (clipping_factors.shape[0],) + (1,) * (g.ndim - 1)
        factors = clipping_factors.reshape(broadcast_shape)
        return g * factors

    per_sample_clipped_grads = tree_map(clip_gradient, per_sample_grads)

    if clip_debug:
        per_sample_clipped_squared_norms = tree_map(squared_l2_norm, per_sample_clipped_grads)
        per_sample_clipped_total_squared_norms = sum(tree_leaves(per_sample_clipped_squared_norms))
        per_sample_clipped_grad_norms = jnp.sqrt(per_sample_clipped_total_squared_norms)
        avg_grad_norm_after = jnp.mean(per_sample_clipped_grad_norms)
        debug.print("Average per-sample gradient norm after clipping (global): {x}", x=avg_grad_norm_after)

    batch_sum_clipped_grads = tree_map(lambda g: jnp.sum(g, axis=0), per_sample_clipped_grads)
    return batch_sum_clipped_grads

compute_batch_global_clipped_grads = jit(compute_batch_global_clipped_grads, static_argnums=(4))


def training_step_global(params, train_data, train_labels, learning_rate, sigma, global_clip_norm, key, batch_size, clip_debug=False):
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size
    accumulated_grads = tree_map(lambda p: jnp.zeros_like(p), params)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        batch_grads = compute_batch_global_clipped_grads(params, batch_data, batch_labels, global_clip_norm, clip_debug)
        accumulated_grads = accumulate_grads(accumulated_grads, batch_grads)

    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)
    keys = tuple(jrandom.split(key, len(tree_leaves(avg_grads))))
    noises = tree_map(
        lambda k, g: jnp.sqrt(learning_rate) * (2 * global_clip_norm / num_samples) * sigma * jrandom.normal(k, g.shape),
        keys, avg_grads
    )
    noisy_avg_grads = tree_map(lambda g, n: g + n, avg_grads, noises)
    new_params = tree_map(lambda p, g: p - learning_rate * g, params, noisy_avg_grads)
    return new_params

training_step_global = jit(training_step_global, static_argnums=(7, 8))


# ----------------------------------------------------------------------


@jit
def loss(params, x, target):
    pred = predict(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(pred, target)


@jit
def evaluate(params, data, labels):
    predictions = predict(params, data)
    predicted_labels = jnp.argmax(predictions, axis=1)
    accuracy = jnp.mean(predicted_labels == labels)
    return accuracy


def train(params, learning_rate, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, sigma, clip_norms, print_every, clipping_mode='global', clip_debug=False):
    key = jrandom.PRNGKey(0)
    metrics = {'train_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        key, subkey = jrandom.split(key)
        
        if clipping_mode == 'layer':
            params = training_step_full(params, train_data, train_labels, learning_rate, sigma, clip_norms, subkey, batch_size, clip_debug)
        elif clipping_mode == 'global':
            global_clip_norm = clip_norms[0]
            params = training_step_global(params, train_data, train_labels, learning_rate, sigma, global_clip_norm, subkey, batch_size, clip_debug)
        else:
            raise ValueError("Invalid training mode")

        train_loss = jnp.mean(loss(params, train_data, train_labels))
        test_acc = evaluate(params, test_data, test_labels)
        
        metrics['train_loss'].append(train_loss.item())
        metrics['test_acc'].append(test_acc.item())
        
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

    return metrics
