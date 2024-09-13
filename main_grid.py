import numpy as np
import argparse
import os
import time
import json
import sys

def relu(a):
    vec_relu = np.vectorize(lambda x: x * (x > 0))
    return vec_relu(a)

def tanh(a):
    vec_tanh = np.vectorize(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
    return vec_tanh(a)


parser = argparse.ArgumentParser()
parser.add_argument('--k')
parser.add_argument('--dataset')
args = parser.parse_args()

k = int(args.k)

time.sleep(k)

save_dir = os.path.join(args.dataset, '30_08_grid')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename = f'parameters_{k}.txt'
file_path = os.path.join(save_dir, filename)
if os.path.exists(file_path):
    print(f"File '{filename}' already exists. Skipping computation for k={k}.")
    sys.exit(0)

print(f"Starting computation for k={k}...")

phi = tanh

if args.dataset == 'synthetic':
    d = 100
    
u = np.random.randn(d) / np.sqrt(d)
target = 'sign'

n = 2000
p = 40000

nt = 1000

epss = [1, 2, 4]
eps = epss[k % 3]
k = k // 3

Cs = np.logspace(-2, 1.5, 20)
C = Cs[k % 20]
k // 20

CTs = np.logspace(-1.5, 2.5, 20)

for CT in CTs:

    delta = 1 / n
    eta = 1 / p
    clip = C * np.sqrt(p)
    tau = CT * d / p
    T = int(tau / eta)
    
    sigma = np.sqrt(eta * T) * np.sqrt(8 * np.log(1 / delta)) / eps

    if args.dataset == 'synthetic':

        X = np.random.randn(n, d)
        Xt = np.random.randn(nt, d)
        
        if target == 'sign':
            Y = np.sign(X @ u)
            Yt = np.sign(Xt @ u)
        elif target == 'linear':
            Y = X @ u
            Yt = Xt @ u

    V = np.random.randn(p, d) / np.sqrt(d)
    
    Phi = phi(X @ V.transpose())
    Phit = phi(Xt @ V.transpose())

    theta = np.zeros(p)

    for t in range(T):
        
        gradients = []
        
        for i in range(n):
            gradient_i = 2 * Phi[i] * (Phi[i].dot(theta) - Y[i])
            norm_i = np.linalg.norm(gradient_i)
            if norm_i > clip:
                gradient_i = (clip / norm_i) * gradient_i
            gradients.append(gradient_i)
        
        gradients = np.array(gradients)
        g_theta = np.mean(gradients, axis=0)

        
        noise = np.sqrt(eta) * (2 * clip / n) * sigma * np.random.normal(0, 1, size=p)
        theta = theta - eta * g_theta + noise

    
    theta_T = theta
    final_test_loss = np.linalg.norm(Phit @ theta_T - Yt) ** 2 / nt


    with open(os.path.join(save_dir, 'parameters_' + args.k + '.txt'), 'a') as f:
        f.write(
            f"{d}\t{n}\t{p}\t{final_test_loss}\t"
            f"{target}\t{eps}\t"
            f"{CT:.2f}\t{C:.2f}\n"
        )

