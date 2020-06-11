import torch
import torch.optim as optim

def create_optimizer(model, hparams):
	if hparams['optim_type'] == 'sgd':
		if hparams['momentum'] > 0:
			return optim.SGD(
				[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr': hparams['base_lr']}], 
				momentum=hparams['momentum'],
				weight_decay=hparams['weight_decay'],
				nesterov=True
				)
		else:
			return optim.SGD(
				[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr': hparams['base_lr']}], 
				momentum=hparams['momentum'],
				weight_decay=hparams['weight_decay'],
				nesterov=False
				)
	elif hparams['optim_type'] == 'adam':
		return optim.AdamW(
			[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr' : hparams['base_lr']}],	
			weight_decay=hparams['weight_decay'])

def one_step_optim(model, lr, momentum=0, weight_decay=0, inner_opt=None):
	return optim.SGD(
		[{'params': model.parameters(), 'lr': lr, 'initial_lr' : lr}], 
		momentum=momentum,
		weight_decay=weight_decay,
		nesterov=True if momentum > 0 else False
	)