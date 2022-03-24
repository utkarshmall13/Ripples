import torch
import argparse
import numpy as np
from src.model import Model, NTXentLoss
from src.dataset import main_transform, SentinelDataset, FullDataset, EurosatDataset
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from os.path import isdir, join
from os import mkdir

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
parser.add_argument('--backbone', '-bb', default='r18')
parser.add_argument('--epochs', '-e', default=5, type=int)
parser.add_argument('--batch-size', '-bs', default=512, type=int)
parser.add_argument('--model-type', '-mt', default='ours')
parser.add_argument('--model-dir', '-md', default='models')
parser.add_argument('--change-event-dir', '-ced', default='change_event_slices')
parser.add_argument('--full-data-dir', '-fdd', default='sentinel_cairo')
parser.add_argument('--eurosat-dir', '-ed', default='Eurosat')  # PATH to EUROSAT
args = parser.parse_args()

################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
simclr_criterion = NTXentLoss('cuda', args.batch_size, 0.07, True)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

################################################################################
# functions for saving and loading models
def save_model(model, fname):
	torch.save(model.state_dict(), fname)


def load_model(model, fname):
	model.load_state_dict(torch.load(fname))
################################################################################


if __name__ == "__main__":
	if args.model_type=='ours':
		train_dataset = SentinelDataset(args.change_event_dir, transform=main_transform)
	elif args.model_type=='nochange':
		train_dataset = SentinelDataset(args.change_event_dir, transform=main_transform, change=False)
	elif args.model_type=='fulldata':
		data_dir = args.full_data_dir
		train_dataset = FullDataset(data_dir, transform=main_transform)
	elif args.model_type=='eurosat':
		data_dir = args.eurosat_dir
		train_dataset = EurosatDataset(data_dir, transform=main_transform)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True, drop_last=True)
	# exit()

	for epoch in range(args.epochs):
		epoch_loss = []
		iter_loss = []
		model.train()
		pbar = tqdm(total=len(train_dataloader))
		for ite, (sample1, sample2) in enumerate(train_dataloader):
			optimizer.zero_grad()
			z1, z2 = model(sample1['first'].to(device), sample1['second'].to(device), sample2['first'].to(device), sample2['second'].to(device), sample1['mask'].to(device), sample2['mask'].to(device))
			loss_SIMCLR = simclr_criterion(z1, z2)
			loss_SIMCLR.backward()
			optimizer.step()
			epoch_loss.append(loss_SIMCLR.item())
			iter_loss.append(loss_SIMCLR.item())
			pbar.update(1)
			pbar.write('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, ite+1, np.mean(iter_loss)))
			if ite%1==0:
				print()
				iter_loss = []
		pbar.write('Epoch: {}, Loss: {}'.format(epoch, np.mean(epoch_loss)))

		# model.eval()
		# for ite, (sample1, sample2) in enumerate(train_dataloader):

		if not isdir(args.model_dir):
			mkdir(args.model_dir)
		save_model(model, join(args.model_dir, 'ripplemodel_'+args.model_type+'_'+str(args.learning_rate)+'.pth.tar'))
		exp_lr_scheduler.step()

################################################################################
# evaluation on eurosat dataset
