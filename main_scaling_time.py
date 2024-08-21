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

save_dir = os.path.join(args.dataset, '21_08_scaling_time_fine')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = tanh  # changed!

if args.dataset == 'synthetic':  # only option now
    # ds = [100, 200]
    # d = ds[k % 2]  # 2 options
    # k = k // 2
    d = 50
    
u = np.random.randn(d) / np.sqrt(d)
# targets = ['linear', 'sign']
# target = targets[k % 2]  # 2 options
# k = k // 2
target = 'sign'

ns = [4000]  # saving the json now works because we have only one n
ps = list(range(15000, 30001, 1000))

nt = 1000

# epss = [1, 2, 4]
# eps = epss[k % 3]  # 3 options
# k = k // 3

eps = 4
eta = 1e-4

'''
Before I had
tau = CT * d / p
T = int(tau / eta)

Now, I still want T eta = CT_old * d / p
1 / (p eta) \sim 0.5

I need new CT like half of the old ones.
'''

# Cs = [0.2, 0.5, 1]
# C = Cs[k % 3]  # 10 options
# k = k // 3

C = 0.5

CTs = np.logspace(np.log10(1), np.log10(50), num=30)
CT = CTs[k % 30]  # 30 options
k = k // 30

for n in ns:
    for p in ps:
        
        data = {}

        train_losses = []
        test_losses = []
        grad_norms = []  # List to store average gradient norms

        delta = 1 / n
        clip_counts = [0] * n
        
        # etas = [1 / (p), 1 / (p * n)]
          # etas[k % 2]  # Let's try

        clip = C * np.sqrt(p)  # * np.log(n)  # ** 2 I remove the log ** 2 for now... logs are not constants in real life
        tau = CT * d / p  # * np.log(n) / p  # **2 same here!
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
        
        Phi = phi(X @ V.transpose())  # using ReLU at the moment
        Phit = phi(Xt @ V.transpose())

        K_inv = np.linalg.inv(Phi @ Phi.transpose())
        pPhi = Phi.transpose() @ K_inv
        theta_star = pPhi @ Y
        # DP-GD
    
        theta = np.zeros(p)  # theta_0
        agg_noise = np.zeros(p)

        for t in range(T):
            # print(t, flush=True)

            # if t % (T // 10) == 0:  # T should be larger than 100
            train_loss = np.linalg.norm(Phi @ theta - Y) ** 2 / n
            test_loss = np.linalg.norm(Phit @ theta - Yt) ** 2 / nt
            # print('losses', train_loss, test_loss, flush=True)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            gradients = []
            batch_grad_norms = []
            
            for i in range(n):
                gradient_i = 2 * Phi[i] * (Phi[i].dot(theta) - Y[i])
                norm_i = np.linalg.norm(gradient_i)
                if norm_i > clip:
                    gradient_i = (clip / norm_i) * gradient_i
                    clip_counts[i] += 1
                # if t % (T // 100) == 0 and i % 10 == 0:
                #     print(norm_i / np.sqrt(p), clip / np.sqrt(p))
                
                gradients.append(gradient_i)
                batch_grad_norms.append(norm_i)  # Store gradient norm
            
            gradients = np.array(gradients)
            g_theta = np.mean(gradients, axis=0)

            avg_grad_norm = np.mean(batch_grad_norms)  # Calculate average gradient norm for this iteration
            grad_norms.append(avg_grad_norm)  # Save average gradient norm
            # if t % (T // 100) == 0:
            #     print(np.linalg.norm(g_theta))
            
            noise = np.sqrt(eta) * (2 * clip / n) * sigma * np.random.normal(0, 1, size=p)  # sqrt ( 1 / p ) * sqrt ( p ) / n * sqrt (d / p)
            agg_noise += noise
            test_noise_loss = np.linalg.norm(Phit @ agg_noise) ** 2 / nt
            # if t % (T // 10) == 0:  # T should be larger than 100
            #     print('noise loss:', test_noise_loss)
            
            theta = theta - eta * g_theta + noise

            # if t % (T // 10) == 0:  # T should be larger than 100
            #     print('norms:', np.linalg.norm(eta * g_theta), np.linalg.norm(noise))
        
        theta_T = theta
            
        final_train_loss = np.linalg.norm(Phi @ theta_T - Y) ** 2 / n
        final_test_loss = np.linalg.norm(Phit @ theta_T - Yt) ** 2 / nt
        train_loss_star = np.linalg.norm(Phi @ theta_star - Y) ** 2 / n
        test_loss_star = np.linalg.norm(Phit @ theta_star - Yt) ** 2 / nt

        
        data = {
            'd': d,
            'n': n,
            'p': p,
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'train_loss_star': train_loss_star,
            'test_loss_star': test_loss_star,
            'target': target,
            'eps': eps,
            'eta': eta,
            'T': T,
            'C_clip': C,
            'C_time': CT
        }

        # with open(os.path.join(save_dir, 'parameters_' + args.k + '.txt'), 'a') as f:
        #     f.write(str(d) + '\t' + str(n) + '\t' + str(p) + '\t' + str(final_train_loss) + '\t' + str(final_test_loss) + '\t' + target + '\t' + str(eps) + '\t' + str(eta) + '\t' + str(T) + '\t' + str(C) + '\n')

        with open(os.path.join(save_dir, f'data_n={n}_p={p}_' + args.k + '.json'), 'w') as f:
            json.dump(data, f)
