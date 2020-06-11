import torch
import random
import numpy as np
import argparse
import csv
import os

parser = argparse.ArgumentParser(description='quadratically overparameterized model')
parser.add_argument('--d', type=int, default=100, help='dimension')
parser.add_argument('--rho', type=int, default=5, help='sparsity of the ground truth weight')
parser.add_argument('--m', type=int, default=40, help='number of examples')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 100)')
parser.add_argument('--alpha', type=float, default=1.0, help='scale of initialization')
parser.add_argument('--delta', type=float, default=1.0, help='scale of noise in label noise / sgd noise / Gaussian noise')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--iter', type=int, default=300000, help='number of iterations')
parser.add_argument('--opt', type=str, default="SGD", choices=["GD", "SGD", "SGLD", "LNGD"], help='type of optimizer')
parser.add_argument('--bs', type=int, default=1, help='Batch size (only useful for SGD)')
parser.add_argument('--log_interval', type=int, default=20, help='how often to update the log')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Logging file
dir_name = "SparseRegression-d:{}-rho:{}-m:{}-seed:{}-alpha:{}-delta:{}-lr:{}-iter:{}-opt:{}-bs:{}-log_interval:{}".format(args.d, args.rho, args.m, args.seed, args.alpha, args.delta, args.lr, args.iter, args.opt, args.bs, args.log_interval)
if not os.path.exists(os.path.join('log', dir_name)):
    os.makedirs(os.path.join('log', dir_name))
log_file = open(os.path.join('log', dir_name, 'log.csv'), mode='w')
fieldnames = ['iter', 'train_error', 'test_error']
log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
log_writer.writeheader()

# Generate initialization matrix
v_0 = np.ones(args.d) * args.alpha

# Generate ground truth weight
w_true = np.random.rand(args.d)
for i in range(args.d):
    w_true[i] = 0.0
for i in range(args.rho):
    w_true[i] = 1.0

# Generate the dataset
dataset = []
labelset = []
xs = np.random.randn(args.m, args.d)
for i in range(args.m):
    x = xs[i, :]
    y = np.inner(x, w_true)
    dataset.append(torch.tensor(x, dtype=torch.float, device="cpu", requires_grad=False))
    labelset.append(torch.tensor(y, dtype=torch.float, device="cpu", requires_grad=False))

# Calculate training error
def cal_train_error(v):
    sum = 0.0
    for i in range(args.m):
        sum += 0.25 * (labelset[i] - np.inner(np.multiply(v, v), dataset[i])) ** 2
    return sum/args.m

# Calculate test error
def cal_test_error(v):
    return np.linalg.norm(np.multiply(v, v) - torch.tensor(w_true, dtype=torch.float, device="cpu", requires_grad=False))**2

# Full gradient descent
def train_GD():
    v = torch.tensor(v_0, dtype=torch.float, device="cpu", requires_grad=True)
    for i in range(args.iter):
        if v.grad is not None:
            v.grad.zero_()
        output = 0
        for idx in range(args.m):
            output += 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.m
        output.backward()
        if i%args.log_interval==0:
            train_error = cal_train_error(v.data)
            test_error = cal_test_error(v.data)
            print("iter: {:3d} | "
                  "train error: {:4.6f} | "
                  "test error: {:4.6f} | "
            .format(
                i,
                train_error,
                test_error,
            ))
            log_writer.writerow({'iter': i, 'train_error': train_error.item(), 'test_error': test_error})
            log_file.flush()
        v.data -= args.lr * v.grad

# Gradient descent with mini-batch noise
def train_SGD():
    v = torch.tensor(v_0, dtype=torch.float, device="cpu", requires_grad=True)
    for i in range(args.iter):
        if v.grad is not None:
            v.grad.zero_()
        output = 0
        for idx in range(args.m):
            output += 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.m
        idxs = np.random.choice(len(dataset), args.bs, replace=False)
        for idx in idxs:
            output += args.delta * 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.bs
        idxs = np.random.choice(len(dataset), args.bs, replace=False)
        for idx in idxs:
            output -= args.delta * 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.bs
        output.backward()
        if i%args.log_interval==0:
            train_error = cal_train_error(v.data)
            test_error = cal_test_error(v.data)
            print("iter: {:3d} | "
                  "train error: {:4.6f} | "
                  "test error: {:4.6f} | "
            .format(
                i,
                train_error,
                test_error,
            ))
            log_writer.writerow({'iter': i, 'train_error': train_error.item(), 'test_error': test_error})
            log_file.flush()
        v.data -= args.lr * v.grad

# Gradient descent with label noise
def train_LNGD():
    v = torch.tensor(v_0, dtype=torch.float, device="cpu", requires_grad=True)
    for i in range(args.iter):
        if i==100000 or i==200000:
            args.lr /= 10.0
        if v.grad is not None:
            v.grad.zero_()
        idxs = np.random.choice(len(dataset), 1, replace=False)
        output = 0
        for idx in range(args.m):
            output += 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.m
        for idx in idxs:
            output += 0.25 * ((labelset[idx] + np.random.normal() * args.delta - torch.dot(torch.mul(v, v), dataset[idx])) ** 2)
            output -= 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2)
        output.backward()
        if i%args.log_interval==0:
            train_error = cal_train_error(v.data)
            test_error = cal_test_error(v.data)
            print("iter: {:3d} | "
                  "train error: {:4.6f} | "
                  "test error: {:4.6f} | "
            .format(
                i,
                train_error,
                test_error,
            ))
            log_writer.writerow({'iter': i, 'train_error': train_error.item(), 'test_error': test_error})
            log_file.flush()
        v.data -= args.lr * v.grad

# Gradient descent with spherical Gaussian noise
def train_SGLD():
    v = torch.tensor(v_0, dtype=torch.float, device="cpu", requires_grad=True)
    for i in range(args.iter):
        if v.grad is not None:
            v.grad.zero_()
        output = 0
        for idx in range(args.m):
            output += 0.25 * ((labelset[idx] - torch.dot(torch.mul(v, v), dataset[idx])) ** 2) / args.m
        output.backward()
        if i%args.log_interval==0:
            train_error = cal_train_error(v.data)
            test_error = cal_test_error(v.data)
            print("iter: {:3d} | "
                  "train error: {:4.6f} | "
                  "test error: {:4.6f} | "
            .format(
                i,
                train_error,
                test_error,
            ))
            log_writer.writerow({'iter': i, 'train_error': train_error.item(), 'test_error': test_error})
            log_file.flush()
        v.data -= args.lr * (v.grad + args.delta * torch.normal(0.0, 1.0, size=(args.d,)))

if args.opt == "SGD":
    train_SGD()
elif args.opt == "LNGD":
    train_LNGD()
elif args.opt == "SGLD":
    train_SGLD()
elif args.opt == "GD":
    train_GD()
