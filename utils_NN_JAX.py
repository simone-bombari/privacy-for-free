import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, nn, debug
import optax
import jax.random as jrandom
from jax.tree_util import tree_map, tree_leaves
from torchvision.datasets import MNIST
import numpy as np
from jax.flatten_util import ravel_pytree


def load_mnist(n, key, data_dir='./data'):
    mnist_train = MNIST(root=data_dir, train=True, download=False)
    mnist_test = MNIST(root=data_dir, train=False, download=False)

    X_train_full = np.array([np.array(image) for image in mnist_train.data], dtype=np.float32) / 255.0
    y_train_full = np.array(mnist_train.targets, dtype=int)
    X_test = np.array([np.array(image) for image in mnist_test.data], dtype=np.float32) / 255.0
    y_test = np.array(mnist_test.targets, dtype=int)

    X_train_full = X_train_full.reshape((X_train_full.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    total_train_samples = X_train_full.shape[0]
    indices = jrandom.permutation(key, total_train_samples)
    X_train_full_shuffled = X_train_full[indices]
    y_train_full_shuffled = y_train_full[indices]

    X_train = X_train_full_shuffled[:n]  # Use the first n samples for training
    y_train = y_train_full_shuffled[:n]
    X_val = X_train_full_shuffled[-10000:]  # Use the last 10000 samples for validation (train with at most 50000 then...)
    y_val = y_train_full_shuffled[-10000:]

    train_data = jnp.array(X_train)
    val_data = jnp.array(X_val)
    test_data = jnp.array(X_test)
    train_labels = jnp.array(y_train)
    val_labels = jnp.array(y_val)
    test_labels = jnp.array(y_test)

    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)

    train_data = (train_data - mean) / (std + 1e-8)
    val_data = (val_data - mean) / (std + 1e-8)
    test_data = (test_data - mean) / (std + 1e-8)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# ----------------------------------------------------------------------

def initialize_params(input_dim, hidden_dim, output_dim, key):
    key1, key2 = jrandom.split(key)
    stddev1 = jnp.sqrt(2.0 / input_dim)
    stddev2 = jnp.sqrt(2.0 / hidden_dim)
    V_1 = jrandom.normal(key1, (hidden_dim, input_dim)) * stddev1
    V_2 = jrandom.normal(key2, (output_dim, hidden_dim)) * stddev2
    return (V_1, V_2)
initialize_params = jit(initialize_params, static_argnums=(0, 1, 2)) 

@jit
def predict(params, x):
    V_1, V_2 = params
    pre_act = jnp.matmul(x, V_1.T)
    act = nn.relu(pre_act)
    out = jnp.matmul(act, V_2.T)
    return out

# ----------------------------------------------------------------------

# these two are used only in the train loop (last function of the file)

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

# ----------------------------------------------------------------------

@jit
def compute_batch_clipped_grads(params, batch_data, batch_labels, clip_norm):
    
    def single_sample_loss(params, x, y):
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)
    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))  # this is like in GD
    
    batch_grads = batch_grad_fn(params, batch_data, batch_labels)  # and this is the line we had in process_batch in GD

    '''
    The following function clips the full per sample gradients
    We do not use layer-wise clipping, as we simply apply DP-GD to the two stacked gradients
    '''
    def clip_gradient(per_sample_grad, clip_norm):
        flat_grad, unravel_fn = ravel_pytree(per_sample_grad)
        grad_norm = jnp.linalg.norm(flat_grad)
        scale = jnp.minimum(1.0, clip_norm / grad_norm)
        flat_clipped_grad = flat_grad * scale
        clipped_grad = unravel_fn(flat_clipped_grad)
        return clipped_grad
    
    batch_clipped_grads = vmap(clip_gradient, in_axes=(0, None))(batch_grads, clip_norm)

    return batch_clipped_grads


@jit
def training_step(params, batched_data, batched_labels, learning_rate, sigma, clip_norm, key, num_samples):

    def process_batch(carry, batch_data_and_labels):
        batch_data, batch_labels = batch_data_and_labels
        batch_clipped_grads = compute_batch_clipped_grads(params, batch_data, batch_labels, clip_norm)  # differs from GD implementation
        batch_sum_clipped_grads = tree_map(lambda g: jnp.sum(g, axis=0), batch_clipped_grads)
        new_carry = tree_map(lambda acc, batch: acc + batch, carry, batch_sum_clipped_grads)
        return new_carry, None

    init_carry = tree_map(lambda p: jnp.zeros_like(p), params)  # my init carry for scan
    
    accumulated_grads, _ = jax.lax.scan(
        process_batch, init_carry, (batched_data, batched_labels)
    )
    
    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)

    num_leaves = len(tree_leaves(avg_grads))  # differs from GD implementation as we need to generate noise now...
    keys = jrandom.split(key, num_leaves)
    key_tree = jax.tree_unflatten(jax.tree_structure(avg_grads), keys)

    '''
    Notice that we are not using sqrt(lr) here
    That term is missing also in the definition of sigma in the main file
    This is justified below, as the noise is also multiplied by lr
    This is closer to DP-GD as written in Abadi&al.2016
    '''
    noise_scale = (2 * clip_norm / num_samples) * sigma
    noises = tree_map(
        lambda k, g: noise_scale * jrandom.normal(k, g.shape),
        key_tree, avg_grads
    )
    noisy_avg_grads = tree_map(lambda g, n: g + n, avg_grads, noises)  # lr multiplies the noises as well!
    
    new_params = tree_map(lambda p, g: p - learning_rate * g, params, noisy_avg_grads)
    return new_params


def train(params, learning_rate, train_data, train_labels, val_data, val_labels, test_data, test_labels, num_epochs, batch_size, sigma, clip_norm, print_every, key):
    metrics = {'train_losses': [], 'val_accuracies': [], 'test_accuracies': []}

    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size
    batched_data = train_data[:num_batches * batch_size].reshape((num_batches, batch_size, -1))
    batched_labels = train_labels[:num_batches * batch_size].reshape((num_batches, batch_size))

    for epoch in range(num_epochs):
        key, subkey = jrandom.split(key)
        
        params = training_step(params, batched_data, batched_labels, learning_rate, sigma, clip_norm, subkey, num_samples)
        
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            train_loss = jnp.mean(loss(params, train_data, train_labels))
            val_acc = evaluate(params, val_data, val_labels)
            test_acc = evaluate(params, test_data, test_labels)
            print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Val accuracy = {val_acc:.4f}")
            metrics['train_losses'].append(train_loss.item())
            metrics['val_accuracies'].append(val_acc.item())
            metrics['test_accuracies'].append(test_acc.item())

    return metrics