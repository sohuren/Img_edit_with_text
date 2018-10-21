import time
import os
from options.demo_options import DemoOptions
opt = DemoOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
import torch
import pickle
import torchvision.transforms as transforms
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.Vgg16 import Vgg16Part
from util.util import init_vgg16
from util.build_vocab import Vocabulary
import torch
from PIL import Image
import nltk
import json
from util import util

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle
opt.lambda_p = 0

# Load vocabulary wrapper.
with open(opt.vocab_path, 'rb') as f:
     vocab = pickle.load(f)

opt.vocab = vocab
opt.vocab_size = len(vocab)

print('load vgg16 models')
init_vgg16("vgg_model")
vgg_model = Vgg16Part()
vgg_model.load_state_dict(torch.load('vgg_model/vgg16.weight'))
if torch.cuda.is_available():
   vgg_model.cuda()
opt.vgg_model = vgg_model

model = create_model(opt)

# open the image
img = Image.open(opt.image).convert('RGB') 
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

img = transform(img)

# nltk the description
# Convert caption (string) to word ids.
tokens = nltk.tokenize.word_tokenize(opt.description.lower())
caption = []
caption.extend([vocab(token) for token in tokens])
target_des = torch.Tensor(caption)

# load the caption
description_buckets = {} 
caption_bucket = opt.bucket_description
if os.path.isfile(caption_bucket):
   with open(caption_bucket) as data_file:
        description_bucket = json.load(data_file)
        # for each different buckets, we have the descriptions
        for item in description_bucket.keys():
            # each bucket might have several descriptions
            description_buckets[item.encode("ascii")] = []
            for des in description_bucket[item]:
                tokens = nltk.tokenize.word_tokenize(str(des).lower())
                caption = []
                caption.extend([vocab(token) for token in tokens])
                target = torch.Tensor(caption)
                description_buckets[item].append(target)

# fill the data now
data = {}
data['A'] = img[None]
data['A_paths'] = opt.image
data['B'] = img[None] # this is fake and we don't use it
data['B_paths'] = opt.image # this is also fake
data['des'] = target_des[None] 
data['des_bucket'] = description_buckets
data['des_neg'] = target_des[None] # this is fake and not used in test time

# set the input
model.set_input(data)
model.test()
visuals = model.get_current_visuals()

# save the image
img_path = opt.saved_image
image_numpy = visuals['fake_B']
util.save_image(image_numpy, img_path)
