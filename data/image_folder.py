################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import json
import nltk
import torch
from numpy.random import randint
import PIL

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# augment the data by randomly cropping left,right,top, bottom corner of the image
def make_dataset(dir, data_augment):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

		if data_augment:
                   images.append(path)  # should take are of the order
		   images.append(path)  
		   images.append(path)
		   images.append(path)
		   images.append(path)
		   images.append(path)
		   images.append(path)
		   images.append(path)  
		   images.append(path)
		   images.append(path)
		   images.append(path)
		   images.append(path)
	
		else:
		   images.append(path)
    return images

# load all the image with data augmentation
def default_loader(path, idx):

    img = Image.open(path).convert('RGB')	
    width = img.size[0]
    height = img.size[1]

    img_left = img.crop((0, 0, width/2, height))
    img_right = img.crop((width/2, 0, width, height))

    width = img_left.size[0]
    height = img_right.size[1]

    if idx == 0: 	
       return img

    if idx == 1: # center cropping	
       
       # get image left       			
       center_x = width/2 	       
       center_y = height/2 
       new_width = (width-256)/2
       new_height = (height-256)/2	

       img_left = img_left.crop((center_x-new_width, center_y-new_height, center_x+new_width, center_x+new_height))
       img_right = img_right.crop((center_x-new_width, center_y-new_height, center_x+new_width, center_x+new_height))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))

       return new_im
     	
    if idx == 2: 

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((0, 0, new_width*2, new_height*2))
       img_right = img_right.crop((0, 0, new_width*2, new_height*2))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im

    if idx == 3: 	

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((0, 256, new_width*2, new_height*2+256))
       img_right = img_right.crop((0, 256, new_width*2, new_height*2+256))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im

    if idx == 4: 	 

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((256, 0, new_width*2+256, new_height*2))
       img_right = img_right.crop((256, 0, new_width*2+256, new_height*2))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im

    if idx == 5: 	

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((256, 256, new_width*2+256, new_height*2+256))
       img_right = img_right.crop((256, 256, new_width*2+256, new_height*2+256))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im


    if idx == 6:
       	 	
       return img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    if idx == 7: # center cropping	
       
       # get image left       			
       center_x = width/2 	       
       center_y = height/2 
       new_width = (width-256)/2
       new_height = (height-256)/2	

       img_left = img_left.crop((center_x-new_width, center_y-new_height, center_x+new_width, center_x+new_height))
       img_right = img_right.crop((center_x-new_width, center_y-new_height, center_x+new_width, center_x+new_height))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))

       return new_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
     	
    if idx == 8: 

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((0, 0, new_width*2, new_height*2))
       img_right = img_right.crop((0, 0, new_width*2, new_height*2))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    if idx == 9: 	

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((0, 256, new_width*2, new_height*2+256))
       img_right = img_right.crop((0, 256, new_width*2, new_height*2+256))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    if idx == 10: 	 

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((256, 0, new_width*2+256, new_height*2))
       img_right = img_right.crop((256, 0, new_width*2+256, new_height*2))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    if idx == 11: 	

       new_width = (width-256)/2
       new_height = (height-256)/2			
       img_left = img_left.crop((256, 256, new_width*2+256, new_height*2+256))
       img_right = img_right.crop((256, 256, new_width*2+256, new_height*2+256))
       new_im = Image.new('RGB', ((new_width)*4, (new_height)*2))
       new_im.paste(img_left, (0,0))
       new_im.paste(img_right, (new_width*2,0))
       return new_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)


# load all the description and bucket description, and all the image
# if we use different bucket models, the bucket description might not be used      
class ImageFolder(data.Dataset):

    def __init__(self, root, caption, caption_bucket, vocab, data_augment, transform=None, return_paths=False,
                 loader=default_loader):

        imgs = make_dataset(root, data_augment)

	# open the captions
	with open(caption) as data_file:
	     description = json.load(data_file)


	# initialize with blank
	self.description_bucket = {}
	# open the descriptions for each bucket
	if os.path.isfile(caption_bucket):
	   with open(caption_bucket) as data_file:
	        description_bucket = json.load(data_file)

	   # for each different buckets, we have the descriptions
	   for item in description_bucket.keys():
	       # each bucket might have several descriptions
	       self.description_bucket[item.encode("ascii")] = []	    	    		
	       for des in description_bucket[item]:
	           tokens = nltk.tokenize.word_tokenize(str(des).lower())
                   caption = []
                   caption.extend([vocab(token) for token in tokens])
                   target = torch.Tensor(caption)
	           self.description_bucket[item].append(target) 


        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

	# now the format is changed to the "positive pair" and "negative pair" 
	self.description = description['positive']
	self.description_neg = description['negative']

	# this will duplicates for many times
	self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
	self.vocab = vocab
	self.data_augment = data_augment
	self.idx = 0
		
    # given the index, output the image
    def __getitem__(self, index):

        path = self.imgs[index]
	# get descriptions
	split = [x for x, v in enumerate(path) if v == '/']	 
	index = split[-1]
	img_name = path[index+1:]

	# this is the real image pair	
	des = self.description[img_name]
	# this is the fake image pair 
	des_neg = self.description_neg[img_name]		

	# copy it for multiple time
	des_bucket = self.description_bucket
 
        img = self.loader(path, self.idx)

	if self.data_augment: 	
	   self.idx += 1
	   if self.idx == 12:
	      self.idx = 0	

	# Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(des).lower())
        caption = []
        caption.extend([self.vocab(token) for token in tokens])
        target = torch.Tensor(caption)	

	# convert negative caption to word idx
	tokens = nltk.tokenize.word_tokenize(str(des_neg).lower())
        caption = []
        caption.extend([self.vocab(token) for token in tokens])
        target_neg = torch.Tensor(caption)	


        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path, target, des_bucket, target_neg
        else:
            return img, target, des_bucket, target_neg

    def __len__(self):
        return len(self.imgs)
