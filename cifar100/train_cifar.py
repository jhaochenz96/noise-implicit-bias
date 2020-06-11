import argparse
import os
import shutil
import copy
from datetime import datetime

import torch 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn

import lr_util
import train_util
from train_util import accuracy
import update_loss_util
import optim_util

import models
import pickle

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Testing dropout on vision tasks.')
parser.add_argument('--epochs', default=200, type=int,
					help='number of total epochs to run')
parser.add_argument('--train_by_iters', action='store_true', 
					help='if true, we train by the total number of iters in iters field')
parser.add_argument('--iters', default=78200, type=int, 
					help='number of training iterations.')
parser.add_argument('--optim_type', 
					choices=['sgd', 'adam'],
					default='sgd')
parser.add_argument('--start_epoch', default=1, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('--start_iter', default=1, type=int,
					help='which iteration to start on.')
parser.add_argument('--batch_size', default=128, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
					help='print frequency (default: 10)')
parser.add_argument('--save_freq', type=int, default=50, 
					help='How often to save the model.')
parser.add_argument('--dataset', choices=["cifar10", "cifar100"], default="cifar10",
					help='cifar10 or cifar100')
parser.add_argument('--label_noise', default=0, type=float,
					help='probability of having label noise.')
parser.add_argument('--ln_sched', 
					choices=['fixed', 'vgg_default'], default='fixed',
					help='schedule of the label noise.')
parser.add_argument('--ln_decay',
					type=float,
					default=0.5,
					help='how much to multiply by when we decay')
parser.add_argument('--arch', choices=model_names, default="wideresnet16",
					help='model architecture:' + ' | '.join(model_names))
parser.add_argument('--no_augment', action='store_true', 
					help='whether to have data augmentation')
parser.add_argument('--parallel', action='store_true', help='Whether to run in parallel.')
parser.add_argument('--data_dir', type=str, help='where the CIFAR data is located.')
parser.add_argument('--no_bn', action='store_true', help='If true, BatchNorm will be off.')

update_loss_util.add_update_args(parser)
lr_util.add_lr_args(parser)

parser.set_defaults(no_augment=False, no_bn=False, no_mult_scale=False, also_flip_labels=False, use_norm_one=False)

def main():
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	augment = not args.no_augment
	train_loader, val_loader = train_util.load_data(
		args.dataset, 
		args.batch_size, 
		dataset_path=args.data_dir,
		augment=augment)
	
	print("=> creating model '{}'".format(args.arch))
	model_args = {
		"num_classes": 10 if args.dataset == "cifar10" else 100	
	}
	model = models.__dict__[args.arch](**model_args)
	print("Device count", torch.cuda.device_count())
	if args.parallel:
		model = nn.DataParallel(model)

	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss().cuda()
	optim_hparams = {
		'base_lr' : args.lr, 
		'momentum' : args.momentum,
		'weight_decay' : args.weight_decay,
		'optim_type' : args.optim_type
	}

	lr_hparams = {
		'lr_sched' : args.lr_sched, 
		'use_iter': args.train_by_iters}

	lr_hparams['iters_per_epoch'] = args.iters_per_epoch if args.iters_per_epoch else 391

	inner_lr_hparams = {
		'lr_sched' : args.inner_anneal,
		'use_iter' : args.train_by_iters}

	inner_lr_hparams['iters_per_epoch'] = args.iters_per_epoch if args.iters_per_epoch else 391

	optimizer = optim_util.create_optimizer(
		model,
		optim_hparams)

	curr_iter = args.start_iter
	epoch = args.start_epoch

	best_val = 0

	inner_opt = optim_util.one_step_optim(
		model, args.inner_lr)
	while True:
		model.train()
		train_acc = train_util.AverageMeter()
		train_loss = train_util.AverageMeter()
		timer = train_util.AverageMeter()
		for i, (input_data, target) in enumerate(train_loader):
					
			lr = lr_util.adjust_lr(
				optimizer,
				epoch,
				curr_iter,
				args.lr,
				lr_hparams)

			inner_lr = lr_util.adjust_lr(
				inner_opt,
				epoch,
				curr_iter,
				args.inner_lr,
				inner_lr_hparams)

			target = target.cuda(async=True)
			input_data = input_data.cuda()

			update_hparams = {
				'update_type' : args.update_type.split('zero_switch_')[-1],
				'inner_lr' : inner_lr[0],
				'use_bn' : not args.no_bn,
				'label_noise' : 0,
				'use_norm_one' : args.use_norm_one
			}

			if args.label_noise > 0:
				label_noise = train_util.label_noise_sched(
					args.label_noise, 
					epoch, 
					curr_iter, 
					args.train_by_iters, 
					args.ln_sched, 
					iters_per_epoch=args.iters_per_epoch,
					ln_decay=args.ln_decay)
				if args.update_type != 'mean_zero_label_noise' or args.also_flip_labels:
					# if it is equal, we don't want to flip the labels
					target = train_util.apply_label_noise(
						target,
						label_noise,
						num_classes=10 if args.dataset == 'cifar10' else 100)

				update_hparams['label_noise'] = label_noise
				
			loss, output, time_taken = update_loss_util.update_step(
				criterion,
				optimizer,
				model,
				input_data,
				target,
				update_hparams)

			prec1 = accuracy(output.data, target, topk=(1,))[0]
			train_loss.update(loss, target.size(0))
			train_acc.update(prec1, target.size(0))
			timer.update(time_taken, 1)
			avg_loss = train_loss.avg
			avg_acc = train_acc.avg

			loss_str = 'Loss '
			loss_str += '{:.4f} (standard)\t'.format(avg_loss)


			if i % args.print_freq == 0:
				log_str = ('Epoch: [{0}][{1}/{2}]\t'

				  'Time {3:.3f}\t {4}'
				  'Prec@1 {5:.3f})').format(
					  epoch, i, len(train_loader), timer.avg, loss_str, avg_acc)
				print(log_str)

			curr_iter += 1

		print("Validating accuracy.")
		val_acc, val_loss = train_util.validate(
			val_loader,
			model,
			criterion,
			epoch,
			print_freq=args.print_freq)

		is_best = val_acc > best_val
		best_val = val_acc if is_best else best_val

		print('Best accuracy: ', best_val)

		epoch += 1
		if args.train_by_iters:
			if curr_iter > args.iters:
				break
		else:
			if epoch > args.epochs:
				break
			
main()
