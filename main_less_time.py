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

save_dir = os.path.join(args.dataset, '12_09_original_new_big_ps')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = tanh

if args.dataset == 'synthetic':  # only option now
    d = 100
    
u = np.random.randn(d) / np.sqrt(d)
target = 'sign'

ns = [2000]  # saving the json now works because we have only one n
ps = np.logspace(4.5 + 0.1842, 4.5 + 0.921, num=5)

nt = 1000

epss = [1, 2, 4]
eps = epss[k % 3]  # 3 options
k = k // 3

Cs = [0.2, 0.5, 1]
C = Cs[k % 3]  # 3 options
k = k // 3

# CTs = [2, 5, 10]
# CT = CTs[k % 3]  # 3 options
# k = k // 3

CT = 0

for n in ns:
    for p in ps:

        p = int(p)
        
        data = {}

        train_losses = []
        test_losses = []
        grad_norms = []  # List to store average gradient norms

        train_losses_star = []
        test_losses_star = []
        

        delta = 1 / n
        eta = 1 / p  # for numerical stability (?)

        clip = C * np.sqrt(p)
        tau = CT * d / p
        T = int(tau / eta)

        T_star = T * int(n / d)
        
        sigma = np.sqrt(eta * T) * np.sqrt(8 * np.log(1 / delta)) / eps  # Proposition 3.2

        if args.dataset == 'synthetic':  # only option for now
    
            X = np.random.randn(n, d)
            Xt = np.random.randn(nt, d)
            
            if target == 'sign':
                Y = np.sign(X @ u)
                Yt = np.sign(Xt @ u)
            elif target == 'linear':
                Y = X @ u
                Yt = Xt @ u

        # random features
    
        V = np.random.randn(p, d) / np.sqrt(d)
        
        Phi = phi(X @ V.transpose())
        Phit = phi(Xt @ V.transpose())

        # K_inv = np.linalg.inv(Phi @ Phi.transpose())
        # pPhi = Phi.transpose() @ K_inv
        pPhi = np.linalg.pinv(Phi)
        theta_pseudo = pPhi @ Y

        
        # DP-GD
    
        theta = np.zeros(p)  # theta_0
        theta_star = np.zeros(p)

        for t in range(T):
            
            train_loss = np.linalg.norm(Phi @ theta - Y) ** 2 / n
            test_loss = np.linalg.norm(Phit @ theta - Yt) ** 2 / nt
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
            gradients = []
            batch_grad_norms = []
            
            for i in range(n):
                gradient_i = 2 * Phi[i] * (Phi[i].dot(theta) - Y[i])
                norm_i = np.linalg.norm(gradient_i)
                if norm_i > clip:
                    gradient_i = (clip / norm_i) * gradient_i
                
                gradients.append(gradient_i)
                batch_grad_norms.append(norm_i)  # store gradient norm to study them later
            
            gradients = np.array(gradients)
            g_theta = np.mean(gradients, axis=0)

            avg_grad_norm = np.mean(batch_grad_norms)  # calculate average gradient norm for this iteration to study it later
            grad_norms.append(avg_grad_norm)

            noise = np.sqrt(eta) * (2 * clip / n) * sigma * np.random.normal(0, 1, size=p)  # Update in Algorithm 1
            theta = theta - eta * g_theta + noise
            

        for t in range(T_star):

            train_loss_star = np.linalg.norm(Phi @ theta_star - Y) ** 2 / n
            test_loss_star = np.linalg.norm(Phit @ theta_star - Yt) ** 2 / nt
            train_losses_star.append(train_loss_star)
            test_losses_star.append(test_loss_star)

            g_theta_star = 2 * Phi.T.dot(Phi.dot(theta_star) - Y) / n
            theta_star = theta_star - eta * g_theta_star


        theta_T = theta
        theta_T_star = theta_star
            
        final_train_loss = np.linalg.norm(Phi @ theta_T - Y) ** 2 / n
        final_test_loss = np.linalg.norm(Phit @ theta_T - Yt) ** 2 / nt

        final_train_loss_star = np.linalg.norm(Phi @ theta_T_star - Y) ** 2 / n
        final_test_loss_star = np.linalg.norm(Phit @ theta_T_star - Yt) ** 2 / nt

        train_loss_pseudo = np.linalg.norm(Phi @ theta_pseudo - Y) ** 2 / n
        test_loss_pseudo = np.linalg.norm(Phit @ theta_pseudo - Yt) ** 2 / nt

        
        data = {
            'd': d,
            'n': n,
            'p': p,
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'final_train_loss_star': final_train_loss_star,
            'final_test_loss_star': final_test_loss_star,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'grad_norms': grad_norms,
            'train_losses_star': train_losses_star,
            'test_losses_star': test_losses_star,
            'train_loss_pseudo': train_loss_pseudo,
            'test_loss_pseudo': test_loss_pseudo,
            'target': target,
            'eps': eps,
            'eta': eta,
            'T': T,
            'C_clip': C,
            'C_time': CT
        }

        with open(os.path.join(save_dir, f'data_n={n}_p={p}_' + args.k + '.json'), 'w') as f:
            json.dump(data, f)
