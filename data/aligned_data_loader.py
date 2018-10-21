import random
import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from pdb import set_trace as st
from builtins import object
from PIL import Image
import math
import pickle

# load the paired data contains "A" and "B"          
class PairedData(object):
    def __init__(self, data_loader, fineSize):
        self.data_loader = data_loader
        self.fineSize = fineSize

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):

        # st()
        AB, AB_paths, des, des_bucket, des_neg = next(self.data_loader_iter)
        # st()
        w_total = AB.size(3)
	w = int(w_total/2)
        h = AB.size(2)

	if self.fineSize > 0:
           w_offset = random.randint(0, max(0, w - self.fineSize - 1))
           h_offset = random.randint(0, max(0, h - self.fineSize - 1))
           A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
           B = AB[:, :, h_offset:h_offset + self.fineSize,
               w + w_offset:w + w_offset + self.fineSize]

	# keep it the same without crop
	else:

           A = AB[:, :, 0:h, 0:w]
           B = AB[:, :, 0:h, w:2*w]
        return {'A': A, 'A_paths': AB_paths, 'B': B, 'B_paths': AB_paths, 'des': des, 'des_bucket': des_bucket, 'des_neg':des_neg}


class AlignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize

	# transform the image data to the tensor range in (0, 1)   
        transform = transforms.Compose([
	    # this is fake, we will delete it later				
            # transforms.Scale((opt.loadSize, opt.loadSize), interpolation=Image.BILINEAR),  # BICUBIC or ANTIALIAS
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        # Dataset A
        dataset = ImageFolder(root=opt.dataroot + '/' + opt.phase, caption = opt.caption, caption_bucket = opt.bucket_description, vocab = opt.vocab, data_augment = opt.augment_data,
                              transform=transform, return_paths=True)


        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads),
	    drop_last=False
            )
            
        self.dataset = dataset
        self.paired_data = PairedData(data_loader, opt.fineSize)

    def name(self):
        return 'AlignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)
