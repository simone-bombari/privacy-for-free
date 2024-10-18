import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax
import argparse
import os
import time
import json


def tanh(a):
    return jnp.tanh(a)


@jit
def compute_clipped_gradients(theta, Phi, Y, clip):
    residuals = Phi @ theta - Y  # shape: (n,)
    per_sample_gradients = 2 * Phi * residuals[:, None]  # shape: (n, p)
    norms = jnp.linalg.norm(per_sample_gradients, axis=1)  # shape: (n,)
    scaling_factors = jnp.minimum(1.0, clip / norms)
    per_sample_gradients = per_sample_gradients * scaling_factors[:, None]
    g_theta = jnp.mean(per_sample_gradients, axis=0)
    return g_theta


# update_theta is authomatically compiled via the scan call, so JITing it should not be necessary...
def update_theta(theta, noise):
    g_theta = compute_clipped_gradients(theta, Phi, Y, clip)
    theta = theta - eta * g_theta + noise
    return theta, None
    
'''
I am returning one more output as lax.scan wants an auxiliary output y as described below
Basically from JAX documentation, with some parts removed

def scan(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
'''

@jit
def train(theta, noises):
    theta, _ = lax.scan(update_theta, theta, noises)
    return theta
    

parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)
time.sleep(k)

save_dir = os.path.join('synthetic', 'save_dir_RF_time')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

key = jrandom.PRNGKey(k) # need to start with this in JAX
key, subkey = jrandom.split(key)

phi = tanh

d = 100
u = jrandom.normal(subkey, shape=(d,)) / jnp.sqrt(d)
target = 'sign'

n = 2000
ps = jnp.logspace(1.5, 5.3, num=20).astype(int)  # same ps as in main_RF
nt = 1000

eps = 4  # fixed here
eta = 0.3 * 1e-4  # independent from p here
C = 0.5  # foxed here
CTs = jnp.logspace(0, 1.2, num=20)  # optimized here

for CT in CTs:
    for p in ps:
        
        delta = 1 / n
        clip = C * jnp.sqrt(p)
        tau = CT * d / p
        T = int(tau / eta)
        sigma = jnp.sqrt(eta * T) * jnp.sqrt(8 * jnp.log(1 / delta)) / eps

        key, subkey = jrandom.split(key)
        X = jrandom.normal(subkey, shape=(n, d))
        key, subkey = jrandom.split(key)
        Xt = jrandom.normal(subkey, shape=(nt, d))

        if target == 'sign':
            Y = jnp.sign(X @ u)
            Yt = jnp.sign(Xt @ u)
        elif target == 'linear':
            Y = X @ u
            Yt = Xt @ u

        key, subkey = jrandom.split(key)
        V = jrandom.normal(subkey, shape=(p, d)) / jnp.sqrt(d)
        Phi = phi(X @ V.T)
        Phit = phi(Xt @ V.T)

        theta = jnp.zeros(p)

        noise_scale = jnp.sqrt(eta) * (2 * clip / n) * sigma  # this is what multiplies the standard Gaussian noise at each update

        '''
        In the following, I generate all the noises from the very beginning.
        This is done to do JIT compilation of the train function
        This can speedup as I am using the lax.scan function in JAX to optimize the training loop
        Notice that noises is (T x p), and it takes less memory than Phi for large models
        '''
        key, subkey = jrandom.split(key)
        noises = noise_scale * jrandom.normal(subkey, shape=(T, p))

        theta = train(theta, noises)

        residuals_test = Phit @ theta - Yt
        final_test_loss = jnp.mean(residuals_test ** 2)

        data = {
            'd': d,
            'n': n,
            'p': int(p),
            'final_test_loss': float(final_test_loss),
            'target': target,
            'eps': eps,
            'eta': eta,
            'T': T,
            'C_clip': C,
            'C_time': float(CT)
        }

        filename = f'data_CT={float(CT)}_p={p}_{args.k}.json'
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(data, f)
