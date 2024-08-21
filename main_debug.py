import numpy as np
import argparse
import os
import time

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--gt')
parser.add_argument('--fmap')
parser.add_argument('--i')
args = parser.parse_args()


time.sleep(int(args.i))

save_dir = os.path.join('synthetic', args.fmap, args.gt)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

phi = relu
dev_phi = dev_relu
mu1 = 0.5

if 'rf' in args.fmap:
    ks = [10000, 20000, 50000, 100000]
elif 'ntk' in args.fmap:
    ks = [20, 50, 100, 200, 500, 1000]

ds = [50, 100, 200, 500]
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

Nt = 1000

for N in Ns:
    for k in ks:
        for d in ds:
            u = np.random.randn(d) / np.sqrt(d)
            X = np.random.randn(N, d)
    
            Xt = np.random.randn(Nt, d)

            if 'sign' in args.gt:
                G = np.sign(X @ u)
                Gt = np.sign(Xt @ u)
            elif 'linear' in args.gt:
                G = X @ u
                Gt = Xt @ u
            elif 'square' in args.gt:
                G = (X @ u) ** 2
                Gt = (Xt @ u) ** 2
        
            if 'rf' in args.fmap:
                V = np.random.randn(k, d) / np.sqrt(d)
                Phi = phi(X @ V.transpose())
                Phi_t = phi(Xt @ V.transpose())
                
                V_inv =  np.linalg.pinv(V.transpose())
                P = V_inv @ V.transpose()
                
                ort_test = (np.eye(k) - P) @ Phi_t[0]

            elif 'ntk' in args.fmap:
                W = np.random.randn(k, d) / np.sqrt(d)
                Phi = rwt(X, dev_phi(X @ W.transpose()))
                Phi_t = rwt(Xt, dev_phi(Xt @ W.transpose()))
                
                Po = np.ones((k, k)) / k
                P = np.kron(np.eye(d), Po)

                ort_test = (np.eye(k * d) - P) @ Phi_t[0]

                
            theta = np.linalg.pinv(Phi) @ G
            theta_1 = P @ theta

            # Scores
    
            # score_test = 0
            # for i in range(Nt):
            #     if np.inner(Phi_t[i], theta) * Gt[i] > 0:
            #         score_test += 1
            # score_test /= Nt
        
            # score_train = 0  # sanity check
            # for i in range(N):
            #     if np.inner(Phi[i], theta) * G[i] > 0:
            #         score_train += 1
            # score_train /= N
        
            # score_test_1 = 0
            # for i in range(Nt):
            #     if np.inner(Phi_t[i], theta_1) * Gt[i] > 0:
            #         score_test_1 += 1
            # score_test_1 /= Nt
        
            # score_train_1 = 0
            # for i in range(N):
            #     if np.inner(Phi[i], theta_1) * G[i] > 0:
            #         score_train_1 += 1
            # score_train_1 /= N

            test_loss = np.linalg.norm(Phi_t @ theta - Gt) ** 2 / Nt
            train_loss = np.linalg.norm(Phi @ theta - G) ** 2 / N
            test_loss_1 = np.linalg.norm(Phi_t @ theta_1 - Gt) ** 2 / Nt
            train_loss_1 = np.linalg.norm(Phi @ theta_1 - G) ** 2 / N
        
            overall_score_test = '\t'.join([str(test_loss), str(test_loss_1)])
            overall_score_train = '\t'.join([str(train_loss), str(train_loss_1)])
            
            with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
                f.write(str(d) + '\t' + str(k) + '\t' + str(N) + '\t' + str(overall_score_train) + '\t' + str(overall_score_test) + '\n')
