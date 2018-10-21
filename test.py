import time
import os
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
import torch
import pickle

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.Vgg16 import Vgg16Part
from util.util import init_vgg16
from util.build_vocab import Vocabulary


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

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
