import numpy as np
import argparse
import os
import time
import json


def relu(a):
    vec_relu = np.vectorize(lambda x: x * (x > 0))
    return vec_relu(a)

def compute_loss(theta, X, V, Y, phi):
    loss = 0
    n = X.shape[0]
    for i in range(n):
        loss += (phi(V @ X[i]).dot(theta) - Y[i]) ** 2 / n
    return loss


parser = argparse.ArgumentParser()
parser.add_argument('--k')
parser.add_argument('--dataset')
args = parser.parse_args()

k = int(args.k)

time.sleep(k)

save_dir = os.path.join(args.dataset, 'trial_thu_0')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = relu

if args.dataset == 'synthetic':  # only option now
    ds = [100, 300]  
    d = ds[k % 3]  # 3 options
    k // 3

u = np.random.randn(d) / np.sqrt(d)
targets = ['linear', 'sign']
target = targets[k % 2]  # 2 options
k // 2

ns = [5000]
ps = [20000, 50000, 100000, 200000]

nt = 1000

epss = [4, 2, 1, 0.5]
eps = epss[k % 4]  # 4 options
k // 4

Cs = [1, 2, 4]
C = Cs[k % 3]  # 3 options

# 72 options now...

train_losses = []
test_losses = []

for n in ns:
    for p in ps:
        print(n, p, flush=True)

        delta = 1 / n
        clip_counts = [0] * n
        
        etas = [1 / (p), 1 / (p * n)]
        eta = etas[k % 2]

        clip = C * np.sqrt(p)  # * np.log(n)  # ** 2 I remove the log ** 2 for now... logs are not constants in real life
        tau = 2 * d / p  # * np.log(n) / p  # **2 same here!
        T = int(tau / eta)
        
        sigma = np.sqrt(eta * T) * 8 * np.sqrt(np.log(1 / delta)) / eps

        if args.dataset == 'synthetic':  # only option for now
    
            X = np.random.randn(n, d)
            Xt = np.random.randn(nt, d)
            
            if target == 'sign':
                Y = np.sign(X @ u)
                Yt = np.sign(Xt @ u)
            elif target == 'linear':
                Y = X @ u
                Yt = Xt @ u

        # Feature Map
    
        V = np.random.randn(p, d) / np.sqrt(d)
        
        # Phi = phi(X @ V.transpose())  # using ReLU at the moment
        # Phit = phi(Xt @ V.transpose())

        # DP-GD
    
        theta = np.zeros(p)  # theta_0
        agg_noise = np.zeros(p)

        for t in range(T):
            # print(t, flush=True)

            if t % (T // 10) == 0:  # T should be larger than 100
                train_loss = compute_loss(theta, X, V, Y, phi)  # np.linalg.norm(Phi @ theta - Y) ** 2 / n
                test_loss = compute_loss(theta, Xt, V, Yt, phi)  # np.linalg.norm(Phit @ theta - Yt) ** 2 / nt
                print('losses', train_loss, test_loss, flush=True)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
            
            avg_gradient = np.zeros(p)
            
            for i in range(n):
                gradient_i = 2 * phi(V @ X[i]) * (phi(V @ X[i]).dot(theta) - Y[i])
                norm_i = np.linalg.norm(gradient_i)
                if norm_i > clip:
                    gradient_i = (clip / norm_i) * gradient_i
                    clip_counts[i] += 1
                # if t % (T // 100) == 0 and i % 10 == 0:
                #     print(norm_i / np.sqrt(p), clip / np.sqrt(p))
                
                avg_gradient += gradient_i / n
            
            # gradients = np.array(gradients)
            # g_theta = np.mean(gradients, axis=0)
            # if t % (T // 100) == 0:
            #     print(np.linalg.norm(g_theta))
            
            noise = np.sqrt(eta) * (2 * clip / n) * sigma * np.random.normal(0, 1, size=p)  # sqrt ( 1 / p ) * sqrt ( p ) / n * sqrt (d / p)
            agg_noise += noise
            test_noise_loss = compute_loss(agg_noise, Xt, V, np.zeros(n), phi)
            if t % (T // 10) == 0:  # T should be larger than 100
                print('noise loss:', test_noise_loss)
            
            theta = theta - eta * avg_gradient + noise

            # if t % (T // 10) == 0:  # T should be larger than 100
            #     print('norms:', np.linalg.norm(eta * g_theta), np.linalg.norm(noise))
        
        theta_T = theta
            
        final_train_loss = compute_loss(theta_T, X, V, Y, phi)
        final_test_loss = compute_loss(theta_T, Xt, V, Yt, phi)

        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'clip_counts': clip_counts
        }

        with open(os.path.join(save_dir, 'losses_and_clips_' + args.k + '.json'), 'w') as f:
            json.dump(data, f)

        with open(os.path.join(save_dir, 'parameters_' + args.k + '.txt'), 'a') as f:
            f.write(str(d) + '\t' + str(n) + '\t' + str(p) + '\t' + str(final_train_loss) + '\t' + str(final_test_loss) + '\t' + target + '\t' + str(eps) + '\t' + str(eta) + '\t' + str(T) + '\t' + str(C) + '\n')
