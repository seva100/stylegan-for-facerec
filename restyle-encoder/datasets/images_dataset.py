from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, demog=None):
		# self.source_paths = sorted(data_utils.make_dataset(source_root))
		# self.target_paths = sorted(data_utils.make_dataset(target_root))
		print(source_root)
		if isinstance(source_root, str) and source_root.endswith('.txt'):
			self.source_paths = sorted(open(source_root).read().splitlines())
		else:
			self.source_paths = sorted(data_utils.make_dataset_from_set_of_dirs(source_root))
		
		if isinstance(target_root, str) and target_root.endswith('.txt'):
			self.target_paths = sorted(open(target_root).read().splitlines())
		else:
			self.target_paths = sorted(data_utils.make_dataset_from_set_of_dirs(target_root))

		if demog is None:
			self.demog = None
		else:
			self.demog = open(demog).read().splitlines()
			self.demog = {line.split(' ')[0]: line.split(' ')[1]
						  for line in self.demog}

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		to_path = self.target_paths[index]

		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		if self.demog is None:
			demog = 2    # mock value
		else:
			print('[ImagesDataset] demog:', demog)
			demog = int(self.demog[from_path])

		# print('[ImagesDataset] from_im shape:', from_im.shape, 'to_im.shape:', to_im.shape)
		# return from_im, to_im, demog
		return from_im, to_im
