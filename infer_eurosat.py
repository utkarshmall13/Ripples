import torch
import argparse
import numpy as np
from src.model import Model
from src.dataset import test_transform, EurosatDataset
from os.path import isdir, join
from os import mkdir
from tqdm import tqdm

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', '-bb', default='r18')
parser.add_argument('--batch-size', '-bs', default=512, type=int)
parser.add_argument('--model-type', '-mt', default='ours')
parser.add_argument('--eurosat-dir', '-ed', default='Eurosat')  # PATH to EUROSAT
parser.add_argument('--save-feature-dir', '-sfd', default='eurofeats')  # PATH where to save features
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
args = parser.parse_args()

################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
################################################################################
# functions for saving and loading models


def save_model(model, fname):
	torch.save(model.state_dict(), fname)


def load_model(model, fname):
	model.load_state_dict(torch.load(fname))
################################################################################


if __name__ == "__main__":
	if args.model_type=='imnet':
		pass
	else:
		load_model(model, 'models/ripplemodel_'+args.model_type+'_'+str(args.learning_rate)+'.pth.tar')

	sentinel_dataset = EurosatDataset(args.eurosat_dir, mode='test', transform=test_transform)
	test_dataloader = torch.utils.data.DataLoader(sentinel_dataset, batch_size=args.batch_size, num_workers=32)

	model.eval()
	feats = []
	fnames = []

	pbar = tqdm(total=len(test_dataloader))
	for ite, (fname, sample) in enumerate(test_dataloader):
		feat = model.forward_single(sample['first'].to(device), sample['second'].to(device), sample['mask'].to(device))
		feat = feat.detach().cpu().numpy()
		# since features from first and second images are concated, and 1st and 2nd images are same, we only use first half
		feat = feat[:, :feat.shape[1]//2]
		feats.append(feat)
		fnames+=fname
		pbar.update(1)

	feats = np.concatenate(feats, axis=0)
	if not isdir(args.save_feature_dir):
		mkdir(args.save_feature_dir)
	np.savez_compressed(join(args.save_feature_dir, 'feats_'+args.model_type+'.npz'), fnames=fnames, feats=feats)
