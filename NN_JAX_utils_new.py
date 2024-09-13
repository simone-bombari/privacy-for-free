import jax.numpy as jnp
from jax import grad, jit, vmap, nn, debug
import optax
import jax.random as jrandom
import jax
from jax.tree_util import tree_map, tree_leaves, tree_unflatten, tree_structure
from torchvision.datasets import MNIST
import numpy as np


def load_mnist(data_dir='./data'):
    # Load the MNIST dataset from local files
    mnist_train = MNIST(root=data_dir, train=True, download=False)
    mnist_test = MNIST(root=data_dir, train=False, download=False)

    # Convert images and labels to NumPy arrays
    X_train = np.array([np.array(image) for image in mnist_train.data], dtype=np.float32)
    y_train = np.array(mnist_train.targets, dtype=int)
    X_test = np.array([np.array(image) for image in mnist_test.data], dtype=np.float32)
    y_test = np.array(mnist_test.targets, dtype=int)

    # Normalize data to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    # Reshape data from (60000, 28, 28) to (60000, 784) to match sklearn's format
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Convert to JAX arrays
    train_data = jnp.array(X_train)
    test_data = jnp.array(X_test)
    train_labels = jnp.array(y_train)
    test_labels = jnp.array(y_test)

    # Normalize using JAX
    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)
    train_data = (train_data - mean) / (std + 1e-8)
    test_data = (test_data - mean) / (std + 1e-8)

    return train_data, train_labels, test_data, test_labels




# @jit(static_argnums=(0, 1, 2))  -> This throws an error!
def initialize_params(input_dim, hidden_dim, output_dim, key):
    key1, key2 = jrandom.split(key)
    
    # He initialization
    stddev1 = jnp.sqrt(2.0 / input_dim)
    stddev2 = jnp.sqrt(2.0 / hidden_dim)

    V_1 = jrandom.normal(key1, (hidden_dim, input_dim)) * stddev1
    V_2 = jrandom.normal(key2, (output_dim, hidden_dim)) * stddev2

    return (V_1, V_2)

initialize_params = jit(initialize_params, static_argnums=(0, 1, 2))  # Don't change the width in the same script!!!! If you don't use static it complains


@jit
def predict(params, x): # When you call this function, x will always be of shape (784,). Until the end of each epoch, where you get an additional dimension with size 60000 or 10000 (train and test set size)
    V_1, V_2 = params
    pre_act = jnp.matmul(x, V_1.T)  # if x is 1 x d, this returns 1 x width
    act = nn.relu(pre_act)
    out = jnp.matmul(act, V_2.T)  # returns 1 x 10. First dimension is batching
    return out

@jit
def loss(params, x, target):
    pred = predict(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(pred, target)


def clip_gradient(g, clip_norm):
    grad_norm = jnp.linalg.norm(g)
    return jnp.where(grad_norm <= clip_norm, g, g * (clip_norm / grad_norm))  # given the single gradient I want to clip, returns it clipped

@jit
def evaluate(params, data, labels):
    predictions = predict(params, data)
    predicted_labels = jnp.argmax(predictions, axis=1)
    accuracy = jnp.mean(predicted_labels == labels)
    return accuracy


@jit
def accumulate_grads(accumulated_grads, batch_grads):  # both the inputs are tuples long 2, with entries with shape (width, 784) and (10, width). No batch here
    return tree_map(lambda acc, batch: acc + batch, accumulated_grads, batch_grads)
# tree_map is used because the variables of the function are nested objects, tuple + arrays



@jit
def compute_batch_clipped_grads(params, x, y, clip_norms):
    
    def single_sample_loss(params, x, y):
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)

    '''
    grad authomatically computes the grad_function of single sample loss with respect to its first parameter. JAX's grad function is designed to handle nested objects. They just need to be "pytree-like" (they can be flattened into a sequence of arrays and then reconstructed). We don't need tree_map as before, where we were doing an element-wise operation on the nested structures.
    vmap is the vectorized map that transforms the grad_function from above to a batched function
    in_axes=(None, 0, 0) means I will not batch the params (1st argument), but x and y are batched in the 1st axis.\
    '''

    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))

    # Now x and y are of shape (batch_size, 784) and (batch_size,), since y is still a number!
    # This means that per_sample_grads is again a 2 long tuple with entries with shape (batch_size, width, 784) and (batch_size, 10, width)
    per_sample_grads = batch_grad_fn(params, x, y)

    # As above, I apply an operation on a nested structure, and tree_map iterates over the elements (g and c) of per_sample_grads and clip_norms simultaneously. Then, vmap allows the clip_gradient function to act on g and c, which are, let's say, the first element of per sample grads (a tensor with shape (batch_size, width, 784)) and the first element of clip_norms (a number). The power of vmap is to do this in parallel along the first dimension of this tensor, which is indeed the batch dimension.
    per_sample_clipped_grads = tree_map(lambda g, c: vmap(clip_gradient, in_axes=(0, None))(g, c),
                                        per_sample_grads, clip_norms)

    # So the output here is still ((2000, 100, 784), (2000, 10, 100))


    # Debugging: Print information for the first few samples
    # def debug_gradients(grads, clipped_grads):
    #     for i in range(min(10, x.shape[0])):
    #         print(f"Sample {i}:")
    #         for j, (g, cg) in enumerate(zip(tree_leaves(grads), tree_leaves(clipped_grads))):
    '''
    tree_leaves flattens a pytree structure into a list of its leaf nodes. If I have something like ((array1, array2), array3), tree_leaves would return [array1, array2, array3].

    For each parameter tensor I print the norm of the original and the clipped gradient for the i-th sample. Useful for debugging
    '''
    #             print(f"  Param {j}:")
    #             print(f"    Original norm: {jnp.linalg.norm(g[i]):.4f}")
    #             print(f"    Clipped norm: {jnp.linalg.norm(cg[i]):.4f}")
    #             print(f"    Clip norm: {clip_norms[j]:.4f}")
    #         print()

    # print("Debugging gradient information:")
    # debug_gradients(per_sample_grads, per_sample_clipped_grads)
    
    # Sum clipped gradients for this batch
    batch_sum_clipped_grads = tree_map(lambda g: jnp.sum(g, axis=0), per_sample_clipped_grads)

    # if batch_idx == 0:
    #     # This one I use for debugging. I want to see at eery epoch the average norm of the per sample clipped gradients (I do this only on the first batch...)
    #     avg_grad_norms = tree_map(lambda g: jnp.mean(jnp.linalg.norm(g, axis=(-2, -1))), per_sample_clipped_grads)
    #     return batch_sum_clipped_grads, avg_grad_norms
    # else:
    return batch_sum_clipped_grads


def training_step_full(params, train_data, train_labels, learning_rate, noise_multiplier, clip_norms, key, batch_size):
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size
    accumulated_grads = tree_map(lambda p: jnp.zeros_like(p), params)  # p tree loops in params and generates the same nested structure full of zeros.
    # first_batch_avg_grad_norms = None
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        batch_grads = compute_batch_clipped_grads(params, batch_data, batch_labels, clip_norms)
        accumulated_grads = accumulate_grads(accumulated_grads, batch_grads)  # these are all trees without batch_size dimension, since we already summed over it at the end of compute_batch_clipped_grads
        
        # if i == 0:
        #     first_batch_avg_grad_norms = avg_grad_norms
    
    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)
    keys = tuple(jrandom.split(key, len(tree_leaves(avg_grads))))  # I am making two keys here... random independent numbers at every iteration?
    noises = tree_map(
        lambda c, k, g: noise_multiplier * c * jrandom.normal(k, g.shape),
        clip_norms, keys, avg_grads
    )
    new_params = tree_map(lambda p, g, n: p - learning_rate * g + n, params, avg_grads, noises)
    return new_params #, first_batch_avg_grad_norms


training_step_full = jit(training_step_full, static_argnums=(7))


@jit
def compute_batch_global_clipped_grads(params, x, y, clip_norm):
    def single_sample_loss(params, x, y):
        pred = predict(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y)

    # Compute per-sample gradients using vectorized mapping
    batch_grad_fn = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))
    per_sample_grads = batch_grad_fn(params, x, y)

    # Compute per-sample gradient norms before clipping
    def squared_l2_norm(g):
        # Sum of squares across all dimensions except the batch dimension
        return jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim)))

    per_sample_squared_norms = tree_map(squared_l2_norm, per_sample_grads)
    per_sample_total_squared_norms = sum(tree_leaves(per_sample_squared_norms))
    per_sample_grad_norms = jnp.sqrt(per_sample_total_squared_norms)

    # Compute clipping factors
    clipping_factors = jnp.minimum(1.0, clip_norm / per_sample_grad_norms)

    # Clip gradients
    def clip_gradient(g):
        # Reshape clipping_factors for broadcasting
        broadcast_shape = (clipping_factors.shape[0],) + (1,) * (g.ndim - 1)
        factors = clipping_factors.reshape(broadcast_shape)
        return g * factors

    per_sample_clipped_grads = tree_map(clip_gradient, per_sample_grads)

    # # Compute per-sample gradient norms after clipping
    # per_sample_clipped_squared_norms = tree_map(squared_l2_norm, per_sample_clipped_grads)
    # per_sample_clipped_total_squared_norms = sum(tree_leaves(per_sample_clipped_squared_norms))
    # per_sample_clipped_grad_norms = jnp.sqrt(per_sample_clipped_total_squared_norms)

    # # Calculate average norms
    # avg_grad_norm_before_clipping = jnp.mean(per_sample_grad_norms)
    # avg_grad_norm_after_clipping = jnp.mean(per_sample_clipped_grad_norms)

    # # Debugging: Print average norms before and after clipping
    # debug.print("Average gradient norm before clipping: {x}", x=avg_grad_norm_before_clipping)
    # debug.print("Average gradient norm after clipping: {x}", x=avg_grad_norm_after_clipping)

    # Sum clipped gradients for this batch
    batch_sum_clipped_grads = tree_map(lambda g: jnp.sum(g, axis=0), per_sample_clipped_grads)

    return batch_sum_clipped_grads


def training_step_global(params, train_data, train_labels, learning_rate, noise_multiplier, global_clip_norm, key, batch_size):
    num_samples = 60000
    num_batches = num_samples // batch_size
    accumulated_grads = tree_map(lambda p: jnp.zeros_like(p), params)
    # first_batch_avg_grad_norm = None

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        batch_grads = compute_batch_global_clipped_grads(params, batch_data, batch_labels, global_clip_norm)
        accumulated_grads = accumulate_grads(accumulated_grads, batch_grads)

    avg_grads = tree_map(lambda g: g / num_samples, accumulated_grads)

    # Adding noise to the averaged gradients
    keys = tuple(jrandom.split(key, len(tree_leaves(avg_grads))))
    noises = tree_map(
        lambda k, g: noise_multiplier * global_clip_norm * jrandom.normal(k, g.shape),
        keys, avg_grads
    )
    noisy_avg_grads = tree_map(lambda g, n: g + n, avg_grads, noises)

    new_params = tree_map(lambda p, g: p - learning_rate * g, params, noisy_avg_grads)
    return new_params # , first_batch_avg_grad_norm


training_step_global = jit(training_step_global, static_argnums=(7))




def train(params, learning_rate, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, noise_multiplier, clip_norms, print_every, training_mode='full'):
    key = jrandom.PRNGKey(0)
    metrics = {'train_loss': [], 'test_acc': []} #, 'grad_1_norms': [], 'grad_2_norms': []}

    for epoch in range(num_epochs):
        key, subkey = jrandom.split(key)
        
        if training_mode == 'full':
            params = training_step_full(params, train_data, train_labels, learning_rate, noise_multiplier, clip_norms, subkey, batch_size)
        # elif training_mode == 'no_noise':
        #     params, avg_grad_norms = training_step_no_noise(params, train_data, train_labels, learning_rate, clip_norms, batch_size)
        # elif training_mode == 'no_clip':
        #     params, avg_grad_norms = training_step_no_clip(params, train_data, train_labels, learning_rate, batch_size)
        # elif training_mode == 'no_clip_with_noise':
        #     params, avg_grad_norms = training_step_no_clip_with_noise(params, train_data, train_labels, learning_rate, noise_multiplier, subkey, batch_size)
        elif training_mode == 'global':
            global_clip_norm = jnp.sqrt(clip_norms[0] ** 2 + clip_norms[1] ** 2) / jnp.sqrt(2)
            params = training_step_global(params, train_data, train_labels, learning_rate, noise_multiplier, global_clip_norm, subkey, batch_size)
        else:
            raise ValueError("Invalid training mode")

        train_loss = jnp.mean(loss(params, train_data, train_labels))
        test_acc = evaluate(params, test_data, test_labels)
        
        metrics['train_loss'].append(train_loss.item())
        metrics['test_acc'].append(test_acc.item())
        
        # if training_mode == 'global':
        #     metrics['grad_1_norms'].append(avg_grad_norms.item())
        # else:
        #     metrics['grad_1_norms'].append(avg_grad_norms[0].item())
        #     metrics['grad_2_norms'].append(avg_grad_norms[1].item())

        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
            # print(f"Average gradient norms: {avg_grad_norms}")

    return params, metrics
