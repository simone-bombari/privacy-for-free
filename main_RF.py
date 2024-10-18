import numpy as np
import argparse
import os
import time
import json


def tanh(a):
    vec_tanh = np.vectorize(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
    return vec_tanh(a)


parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)  # argument to run multiple scripts in parallel
time.sleep(k)

save_dir = os.path.join('synthetic', 'save_dir_RF')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = tanh

d = 100
u = np.random.randn(d) / np.sqrt(d)  # define the task
target = 'sign'

n = 2000  # number of training data
ps = np.logspace(1.5, 5.3, num=20)  # the p-s explored in Fig 2

nt = 1000  # number of test data

# with the following, I do 27 independent trainings in parallel

epss = [1, 2, 4]
eps = epss[k % 3]  # 3 options
k = k // 3

Cs = [0.2, 0.5, 1]
C = Cs[k % 3]  # 3 options
k = k // 3

CTs = [2, 5, 10]
CT = CTs[k % 3]  # 3 options
k = k // 3


for p in ps:

    p = int(p)
    
    data = {}

    train_losses = []
    test_losses = []

    train_losses_star = []
    test_losses_star = []
    
    delta = 1 / n
    eta = 1 / p  # as p increases I might need smaller learning rates for stability

    clip = C * np.sqrt(p)
    tau = CT * d / p
    T = int(tau / eta)
    
    sigma = np.sqrt(eta * T) * np.sqrt(8 * np.log(1 / delta)) / eps  # From Proposition 3.2

    # Here I build the training and test set

    X = np.random.randn(n, d)
    Xt = np.random.randn(nt, d)
    
    if target == 'sign':
        Y = np.sign(X @ u)
        Yt = np.sign(Xt @ u)
    elif target == 'linear':
        Y = X @ u
        Yt = Xt @ u

    # and the corresponding random features

    V = np.random.randn(p, d) / np.sqrt(d)
    
    Phi = phi(X @ V.transpose())
    Phit = phi(Xt @ V.transpose())

    # Here I compute the closed form for the solution of GD \theta^*
    
    pPhi = np.linalg.pinv(Phi)
    theta_star = pPhi @ Y

    theta = np.zeros(p)  # theta_0

    for t in range(T):
        
        train_loss = np.linalg.norm(Phi @ theta - Y) ** 2 / n
        test_loss = np.linalg.norm(Phit @ theta - Yt) ** 2 / nt
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
        gradients = []
        
        for i in range(n):
            gradient_i = 2 * Phi[i] * (Phi[i].dot(theta) - Y[i])
            norm_i = np.linalg.norm(gradient_i)
            if norm_i > clip:
                gradient_i = (clip / norm_i) * gradient_i
            
            gradients.append(gradient_i)
        
        gradients = np.array(gradients)
        g_theta = np.mean(gradients, axis=0)

        noise = np.sqrt(eta) * (2 * clip / n) * sigma * np.random.normal(0, 1, size=p)  # Update as in Algorithm 1
        theta = theta - eta * g_theta + noise

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
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_loss_star': train_loss_star,
        'test_loss_star': test_loss_star,
        'target': target,
        'eps': eps,
        'eta': eta,
        'T': T,
        'C_clip': C,
        'C_time': CT
    }

    with open(os.path.join(save_dir, f'data_p={p}_' + args.k + '.json'), 'w') as f:
        json.dump(data, f)
