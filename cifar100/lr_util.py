lr_sched_choices = [
	'vgg_default', 
	'fixed']

def add_lr_args(parser):
	parser.add_argument('--lr_sched', choices=lr_sched_choices, default='default', help='Learning rate schedule to use.')
	parser.add_argument('--iters_per_epoch', type=int, help='Value to pass in for iters_per_epoch, if train_by_iters is true.')
	parser.add_argument('--inner_anneal', choices=lr_sched_choices, default='default', help='Inner learning rate schedule to use.')
	
def adjust_lr(optimizer, epoch, curr_iter, lr, hparams):
	lr_vals = []
	for param_group in optimizer.param_groups:
		param_group['lr'] = get_adjusted_lr(
			param_group['initial_lr'], 
			epoch, 
			curr_iter, 
			hparams['use_iter'],
			hparams['lr_sched'],
			iters_per_epoch=hparams['iters_per_epoch'])	
		lr_vals.append(param_group['lr'])
	return lr_vals

def get_adjusted_lr(lr, epoch, curr_iter, use_iter, lr_sched, iters_per_epoch=391):
	if lr_sched == 'default':
		sched_func = lr_sched_default
	if lr_sched == 'vgg_default':
		sched_func = lr_vgg_default
	if lr_sched == 'fixed':
		sched_func = lr_fixed
	if lr_sched == 'decay_once':
		sched_func = lr_decay_once
	if lr_sched == 'decay_twice':
		sched_func = lr_decay_twice
	if lr_sched == 'decay_early':
		sched_func = lr_decay_early
	if lr_sched == 'decay_no_reg':
		sched_func = lr_decay_no_reg

	return sched_func(lr, epoch, curr_iter, use_iter, iters_per_epoch=iters_per_epoch)

def lr_fixed(lr, epoch, curr_iter, use_iter, iters_per_epoch=391):
	return lr

def lr_vgg_default(lr, epoch, curr_iter, use_iter, iters_per_epoch=391):
	if use_iter:
		new_lr = lr*(0.1**int(curr_iter >= 150*iters_per_epoch))
		new_lr *= (0.1**int(curr_iter >= 250*iters_per_epoch))
	else:
		new_lr = lr*(0.1**int(epoch >= 150))
		new_lr *= (0.1**int(epoch >= 250))
	return new_lr