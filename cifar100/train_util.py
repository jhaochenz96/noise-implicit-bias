import os
import shutil
import time

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_train_set(dataset, dataset_path="./", augment=True):
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

	if augment:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
								(4,4,4,4),mode='reflect').squeeze()),
			transforms.ToPILImage(),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			])
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize,
			])

	return datasets.__dict__[dataset.upper()](dataset_path, train=True, download=True,
							 transform=transform_train)

def load_data(dataset, batch_size, dataset_path="./", augment=True):
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
	transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize,
			])
	train_set = get_train_set(dataset, dataset_path=dataset_path, augment=augment)

	kwargs = {'num_workers': 1, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=batch_size, shuffle=True, **kwargs)
	val_loader = torch.utils.data.DataLoader(
		datasets.__dict__[dataset.upper()](dataset_path, train=False, transform=transform_test),
		batch_size=batch_size, shuffle=True, **kwargs)
	return train_loader, val_loader

def apply_label_noise(labels, noise_prob, num_classes=10): 
	should_flip = np.random.random(len(labels))
	rand_labels = torch.randint_like(labels, low=0, high=num_classes)
	flip_mask = should_flip < noise_prob
	labels[flip_mask] = rand_labels[flip_mask]
	return labels

def label_noise_sched(noise_prob, epoch, curr_iter, use_iter, sched, iters_per_epoch=391, ln_decay=0.5):
	if sched == 'fixed':
		return ln_fixed(
			noise_prob,
			epoch,
			curr_iter,
			use_iter,
			iters_per_epoch=iters_per_epoch)
	if sched == 'vgg_default':
		return ln_vgg_default(
			noise_prob,
			epoch,
			curr_iter,
			use_iter,
			iters_per_epoch=iters_per_epoch,
			ln_decay=0.5)

def ln_fixed(noise_prob, epoch, curr_iter, use_iter, iters_per_epoch=391):
	return noise_prob

def ln_vgg_default(noise_prob, epoch, curr_iter, use_iter, iters_per_epoch=391, ln_decay=0.5):
	if use_iter:
		prob = noise_prob*(ln_decay**int(curr_iter >= 150*iters_per_epoch))
		prob *= (ln_decay**int(curr_iter >= 250*iters_per_epoch))
	else:
		prob = noise_prob*(ln_decay**int(epoch >= 150))
		prob *= (ln_decay**int(epoch >= 250))
	return prob

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def validate(
	val_loader, 
	model, 
	criterion, 
	epoch,
	print_freq=10
	):
	"""Perform validation on the validation set"""
	batch_time = AverageMeter()
	val_loss = AverageMeter()
	val_acc = AverageMeter()
	
	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input_data, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_data = input_data.cuda()
		input_var = torch.autograd.Variable(input_data)
		target_var = torch.autograd.Variable(target)

		# compute output
		with torch.no_grad():
			output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		val_loss.update(loss.item(), target.size(0))
		val_acc.update(prec1.item(), target.size(0))
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			log_str = ('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {2:.4f})\t'
				  'Prec@1 {3:.3f})').format(
					  i, len(val_loader), val_loss.avg, val_acc.avg, batch_time=batch_time)
			print(log_str)

	print(' * Prec@1 {:.4f}'.format(val_acc.avg))

	return val_acc.avg, val_loss.avg


