import numpy as np
import argparse
import os
import time
import json


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

save_dir = os.path.join(args.dataset, '12_09_scaling_time_new_big_ps_small_eta')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = tanh

if args.dataset == 'synthetic': 
    d = 100
    
u = np.random.randn(d) / np.sqrt(d)

target = 'sign'

ns = [2000]
ps = np.logspace(4.5 + 0.1842, 4.5 + 0.921, num=5) # np.logspace(1, 4.5, num=20)
# [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70] # [100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000] # [10000, 15000, 20000, 30000, 50000, 70000, 100000]

nt = 1000

eps = 4
# eta = 1e-5  # Fixed here and small enough
eta = 1e-4  # Fixed here and small enough
C = 0.5  # Fixed here

# CTs = np.logspace(np.log10(2), np.log10(15), num=20)  # For this C before I had a good CT at 5
CTs = np.logspace(0, 1.5, num=20)
# CT = CTs[k % 30]  # 30 options
# k = k // 30

for CT in CTs:
    for n in ns:
        for p in ps:
            
            p = int(p)
            
            data = {}
            delta = 1 / n
    
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
     
            data = {
                'd': d,
                'n': n,
                'p': p,
                'final_test_loss': final_test_loss,
                'target': target,
                'eps': eps,
                'eta': eta,
                'T': T,
                'C_clip': C,
                'C_time': CT
            }
    
            with open(os.path.join(save_dir, f'data_CT={CT}_n={n}_p={p}_' + args.k + '.json'), 'w') as f:
                json.dump(data, f)
