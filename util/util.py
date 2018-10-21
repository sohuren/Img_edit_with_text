from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy
import os
import collections
import Vgg16
from torch.autograd import Variable
from torch.utils.serialization import load_lua


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# convert the image tensor(float) to the RGB value 
def tensor2im2(image_tensor):
    image = image_tensor[0].data.float()
    image = (image + 1) / 2.0 * 255.0
    (r, g, b) = torch.chunk(image, 3)
    image = torch.cat((b, g, r))
    tensortype = type(image)
    mean = tensortype(image.size())
    mean[0, :, :] = 103.939
    mean[1, :, :] = 116.779
    mean[2, :, :] = 123.680
    image -= mean

    return Variable(image[None])
		 
# Load VGG16 for torch and save
def init_vgg16(model_folder='model'):
    """load the vgg16 model feature"""
    if not os.path.exists(model_folder+'/vgg16.weight'):
       if not os.path.exists(model_folder+'/vgg16.t7'):
	  os.system('wget http://bengxy.com/dataset/vgg16.t7 '+model_folder+'/vgg16.t7')
       vgglua = load_lua(model_folder + '/vgg16.t7')
       vgg = Vgg16.Vgg16Part()
       for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
	   dst[:].data = src[:]  
	   # here comes a bug in pytorch version 0.1.10
	   # change to dst[:].data = src[:]
	   # ref to issue:
	   
       torch.save(vgg.state_dict(), model_folder+'/vgg16.weight')

def idx2text(text_id, vocab):
    "sampled id"	
    text_id = text_id.numpy()[0]	
    sampled_caption = []
    for word_id in text_id:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)

    sentence = ' '.join(sampled_caption)
    return sentence	
 
def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_description(label, text_path):
    fp = open(text_path, "wt+") 
    fp.write(label)
    fp.close()	

def get_description(text_path):
    fp = open(text_path, "r") 
    line = fp.readline().strip()
    fp.close()	
    return line
	
def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
