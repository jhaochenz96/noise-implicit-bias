import math
import torch
import subprocess
import itertools
import numpy as np 
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import optim_util
from train_util import accuracy
import lr_util
# so we can differentiate the noise update

import higher

update_types = [
	'standard',
	'gaussian_drift',
	'mean_zero_label_noise']

def norm_one_gaussian(in_tensor):
	tensor_size = torch.numel(in_tensor)
	return torch.randn_like(in_tensor)/np.sqrt(tensor_size)

def add_update_args(parser):
	parser.add_argument(
		'--update_type', choices=update_types, help='type of the update to make.')
	parser.add_argument(
		'--inner_lr', 
		type=float,
		default=1,
		help='The learning rate on the noise part of the update.')
	parser.add_argument(
		'--also_flip_labels',
		action='store_true',
		help='Also flip the labels in addition to having mean-zero noise.')
	parser.add_argument(
		'--use_norm_one',
		action='store_true',
		help='Whether to use a norm one Gaussian for each parameter, or have constant scaling.')
		

def update_step(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	if hparams["update_type"] == 'standard':
		return standard(
			criterion,
			optimizer,
			model,
			inputs,
			labels,
			hparams)

	if hparams['update_type'] == 'gaussian_drift':
		return gaussian_drift(
			criterion,
			optimizer,
			model,
			inputs,
			labels,
			hparams)

	if hparams['update_type'] == 'mean_zero_label_noise':
		return mean_zero_label_noise(
			criterion,
			optimizer,
			model,
			inputs,
			labels,
			hparams)

def standard(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	
	start_time = time.time()
	output = model(inputs, use_bn=hparams['use_bn'])
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item(), output, time.time() - start_time

def gaussian_drift(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):	

	start_time = time.time()
	output = model(inputs, use_bn=hparams['use_bn'])
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# now add gaussian noise to the parameters
	for ind, param in enumerate(model.parameters()):
		if hparams['use_norm_one']:
			gaussian_noise = norm_one_gaussian(param.data)
		else:
			gaussian_noise = torch.randn_like(param.data)
		param.data.copy_(param.data + hparams['inner_lr']*gaussian_noise)

	return loss.item(), output, time.time() - start_time

def mean_zero_label_noise(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams
	):
	start_time = time.time()
	output = model(inputs, use_bn=hparams['use_bn'])

	def grad_func(grad):
		in_noise = hparams['label_noise']*norm_one_gaussian(grad.data)
		return grad + in_noise

	output.register_hook(grad_func)
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item(), output, time.time() - start_time