import torch
import torch.nn as nn
from torch.autograd import Variable
from pdb import set_trace as st
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################

# intialize the weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.01, 0.02)
        m.bias.data.fill_(0)

# define the generator
def define_G(input_nc, output_nc, ngf, which_model_netG, norm, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_7blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=7, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, 6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

# define the generator
def define_G_FilterBank(input_nc, output_nc, ngf, which_model_netG, norm, num_G, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_7blocks':
        # not implemented yet
        netG = ResnetGenerator_FilterBank(input_nc, output_nc, ngf, norm_layer, n_blocks=7, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        # not implemented yet
        netG = ResnetGenerator_FilterBank(input_nc, output_nc, ngf, norm_layer, 6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator_FilterBank(input_nc, output_nc, 7, ngf, norm_layer, num_G, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator_FilterBank(input_nc, output_nc, 6, ngf, norm_layer, num_G, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

# define the generator that can incoporate the hidden vector coming from lstm
def define_G_Aug(input_nc, output_nc, ngf, num_txt, which_model_netG, norm, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())
	
    if which_model_netG == 'resnet_7blocks':
        netG = ResnetGenerator_Aug(input_nc, output_nc, num_txt, ngf, norm_layer, n_blocks=7, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator_Aug(input_nc, output_nc, num_txt, ngf, norm_layer, 6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator_Aug(input_nc, output_nc, 7, num_txt, ngf, norm_layer, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator_Aug(input_nc, output_nc, 6, num_txt, ngf, norm_layer, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

# define the discriminator that can tell real or fake
def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = define_D(input_nc, ndf, 'n_layers', use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD

# define the rnn which can encode the text to fixed dimension vector
def define_ENcoder(embed_size, hidden_size, vocab_size, visual_size, num_layers, cell, wordvec, gpu_ids=[]):

    Encoder = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    Encoder = EncoderRNN(embed_size, hidden_size, vocab_size, visual_size, num_layers, cell, wordvec)
    
    if len(gpu_ids) > 0:
        Encoder.cuda()

    # apply the uniformly weight initialization, or we can apply the RNN separate initialization   	
    Encoder.init_weights()

    return Encoder

# define the rnn which can classifiy the text into one of the bucket after softmax
def define_ENcoder_Linear(embed_size, hidden_size, vocab_size, label_size, num_layers, cell, wordvec, gpu_ids=[]):

    Encoder = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    Encoder = EncoderRNN_Linear(embed_size, hidden_size, vocab_size, label_size, num_layers, cell, wordvec)
    
    if len(gpu_ids) > 0:
        Encoder.cuda()

    # apply the uniformly weight initialization, or we can apply the RNN separate initialization   	
    Encoder.init_weights()

    return Encoder

# define the resnet, which can encode the image to a fixed dimension vector through pre-trained resnet
def define_ResnetMLP(input_nc, output_nc, ngf, norm, n_blocks, visual_size, gpu_ids=[]):

    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
	
    Encoder = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())

    Encoder = ResnetCNN(input_nc, output_nc, ngf, norm_layer, n_blocks, visual_size, gpu_ids)
    if len(gpu_ids) > 0:
        Encoder.cuda()

    # apply the uniformly weight initialization, or we can apply the RNN separate initialization   	
    Encoder.apply(weights_init)
    return Encoder


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss used in LSGAN.
# It is basically same as MSELoss, but it abstracts away the need to create
# the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf),
                 nn.ELU(inplace=True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ELU(inplace=True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ELU(inplace=True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator_Aug(nn.Module):
    def __init__(self, input_nc, output_nc, num_txt, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator_Aug, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
	self.num_txt = num_txt

	# down sampling
        down_model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf),
                 nn.ELU(inplace=True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            down_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ELU(inplace=True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            down_model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]

	# up sampling model
	up_model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
	    wordvec_dim = (self.num_txt if i ==0 else 0)
            up_model += [nn.ConvTranspose2d(ngf * mult + wordvec_dim, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ELU(inplace=True)]

        up_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        up_model += [nn.Tanh()]

	self.down_model = nn.Sequential(*down_model)
        self.up_model = nn.Sequential(*up_model)

    def forward(self, input, wordvec):

	wordvec = torch.unsqueeze(wordvec, 2)
        wordvec = torch.unsqueeze(wordvec, 3)
	
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:

            hidden_vec = nn.parallel.data_parallel(self.down_model, input, self.gpu_ids)
	    input_shape = hidden_vec.size()	
	    hidden_vec = torch.cat([hidden_vec, wordvec.expand(input_shape[0], self.num_txt, input_shape[2], input_shape[3])], 1)
	    return nn.parallel.data_parallel(self.up_model, hidden_vec, self.gpu_ids)

        else:
	    
            hidden_vec = self.down_model(input)
	    input_shape = hidden_vec.size()
	    hidden_vec = torch.cat([hidden_vec, wordvec.expand(input_shape[0], self.num_txt, input_shape[2], input_shape[3])], 1)
	    return self.up_model(hidden_vec)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer)

    def build_conv_block(self, dim, padding_type, norm_layer):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ELU(inplace=True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block)
        unet_block = UnetSkipConnectionBlock(input_nc, ngf, unet_block,
                                             outermost=True)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator_Aug(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, num_txt, ngf=64,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UnetGenerator_Aug, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)
        # construct unet structure, from bottom to top
	self.total_block = num_downs*2
	self.num_txt = num_txt

	# each one is a sequential model	
	self.add_module("down_" + str(num_downs-1), UnetSkipConnectionBlock_Down_Aug(ngf * 8, ngf * 8, innermost=True))
	self.add_module("up_" + str(num_downs), UnetSkipConnectionBlock_Up_Aug(ngf * 8, ngf * 8 + num_txt, innermost=True)) # assume that the 256+512

        for i in range(num_downs-5):
            self.add_module("down_" + str(4+i), UnetSkipConnectionBlock_Down_Aug(ngf * 8, ngf * 8))
	    self.add_module("up_" + str(self.total_block-5-i), UnetSkipConnectionBlock_Up_Aug(ngf * 8, ngf * 8))

        self.add_module("down_" + str(3), UnetSkipConnectionBlock_Down_Aug(ngf * 4, ngf * 8))
	self.add_module("up_" + str(self.total_block-4), UnetSkipConnectionBlock_Up_Aug(ngf * 4, ngf * 8))

        self.add_module("down_" + str(2), UnetSkipConnectionBlock_Down_Aug(ngf * 2, ngf * 4))
	self.add_module("up_" + str(self.total_block-3), UnetSkipConnectionBlock_Up_Aug(ngf * 2, ngf * 4))

        self.add_module("down_" + str(1),UnetSkipConnectionBlock_Down_Aug(ngf, ngf * 2))
	self.add_module("up_" + str(self.total_block-2),UnetSkipConnectionBlock_Up_Aug(ngf, ngf * 2))

        self.add_module("down_" + str(0),UnetSkipConnectionBlock_Down_Aug(input_nc, ngf, outermost=True))
        self.add_module("up_" + str(self.total_block-1),UnetSkipConnectionBlock_Up_Aug(input_nc, ngf, outermost=True))

    # forward computing, given the text feature and the image feature	
    def forward(self, input, wordvec):

	# x is the output for each layer
	if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
	
	   x = {}
	   input_i = input
	   for i in range(self.total_block/2): 
               x[i] = nn.parallel.data_parallel(self._modules["down_"+str(i)], input_i, self.gpu_ids)
	       input_i = x[i]	

	   # concatenate the hideen state vector from LSTM
	   input_shape = x[self.total_block/2-1].size()
	   wordvec = torch.unsqueeze(wordvec, 2)
	   wordvec = torch.unsqueeze(wordvec, 3)
	   hidden_vec = torch.cat([x[self.total_block/2-1], wordvec.expand(input_shape[0], self.num_txt, input_shape[2], input_shape[3])], 1)
	   x[self.total_block/2] = nn.parallel.data_parallel(self._modules["up_"+str(self.total_block/2)], hidden_vec, self.gpu_ids)

	   for i in range(self.total_block/2+1, self.total_block):
	       input_i = torch.cat([x[i-1], x[self.total_block-i-1]], 1)	 
               x[i] = nn.parallel.data_parallel(self._modules["up_" + str(i)], input_i, self.gpu_ids)

	   return x[self.total_block-1]	

	else:
	  
	   x = {}
	   input_i = input
	   for i in range(self.total_block/2): 
               x[i] = self._modules["down_"+str(i)](input_i)
	       input_i = x[i]	

	   # concatenate the hideen state vector from LSTM
	   input_shape = x[self.total_block/2-1].size()
	   wordvec = torch.unsqueeze(wordvec, 2)
	   wordvec = torch.unsqueeze(wordvec, 3)
	   hidden_vec = torch.cat([x[self.total_block/2-1], wordvec.expand(input_shape[0], self.num_txt, input_shape[2], input_shape[3])], 1)
	   x[self.total_block/2] = self._modules["up_"+str(self.total_block/2)](hidden_vec)

	   for i in range(self.total_block/2+1, self.total_block):
	       input_i = torch.cat([x[i-1], x[self.total_block-i-1]], 1)	 
               x[i] = self._modules["up_" + str(i)](input_i)

	   return x[self.total_block-1]	


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator_FilterBank(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, num_G=5, gpu_ids=[]):
        super(UnetGenerator_FilterBank, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)
        # construct unet structure, from bottom to top
	self.total_block = num_downs*2

	# each one is a sequential model	
	self.add_module("down_" + str(num_downs-1), UnetSkipConnectionBlock_Down_Aug(ngf * 8, ngf * 8, innermost=True))
	self.add_module("up_" + str(num_downs), UnetSkipConnectionBlock_Up_Aug(ngf * 8, ngf * 8, innermost=True)) # assume that the 256+512

	for i in range(num_downs-5):
		self.add_module("down_" + str(4+i), UnetSkipConnectionBlock_Down_Aug(ngf * 8, ngf * 8))
		self.add_module("up_" + str(self.total_block-5-i), UnetSkipConnectionBlock_Up_Aug(ngf * 8, ngf * 8))

	self.add_module("down_" + str(3), UnetSkipConnectionBlock_Down_Aug(ngf * 4, ngf * 8))
	self.add_module("up_" + str(self.total_block-4), UnetSkipConnectionBlock_Up_Aug(ngf * 4, ngf * 8))

	self.add_module("down_" + str(2), UnetSkipConnectionBlock_Down_Aug(ngf * 2, ngf * 4))
	self.add_module("up_" + str(self.total_block-3), UnetSkipConnectionBlock_Up_Aug(ngf * 2, ngf * 4))

	self.add_module("down_" + str(1),UnetSkipConnectionBlock_Down_Aug(ngf, ngf * 2))
	self.add_module("up_" + str(self.total_block-2),UnetSkipConnectionBlock_Up_Aug(ngf, ngf * 2))

	self.add_module("down_" + str(0),UnetSkipConnectionBlock_Down_Aug(input_nc, ngf, outermost=True))
	self.add_module("up_" + str(self.total_block-1),UnetSkipConnectionBlock_Up_Aug(input_nc, ngf, outermost=True))

	# suppose we have 5 different filters
	ker = 3
	pad = (ker-1)/2
	self.filter_bank_size = num_G
	for i in range(self.filter_bank_size):
		self.add_module("filter_" + str(i), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=ker, stride=1, padding=pad))

    # forward computing, given the text feature and the image feature	
    def forward(self, input):

	# x is the output for each layer
	if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
	
	   x = {}
	   input_i = input
	   for i in range(self.total_block/2): 
               x[i] = nn.parallel.data_parallel(self._modules["down_"+str(i)], input_i, self.gpu_ids)
	       input_i = x[i]	

	   hidden = {}    
	   # do the filter across different filter and save it as hidden states
	   for i in range(self.filter_bank_size): 
               hidden[i] = nn.parallel.data_parallel(self._modules["filter_"+str(i)], input_i, self.gpu_ids)
	       	
       # for each hidden state, do the up sampling now
	   up_sample = {}
	   for  k in range(self.filter_bank_size):
	        x[self.total_block/2] = nn.parallel.data_parallel(self._modules["up_"+str(self.total_block/2)], hidden[k], self.gpu_ids)
	        for i in range(self.total_block/2+1, self.total_block):
	            input_i = torch.cat([x[i-1], x[self.total_block-i-1]], 1)	 
	            x[i] = nn.parallel.data_parallel(self._modules["up_" + str(i)], input_i, self.gpu_ids)
	        up_sample[k] = x[self.total_block-1]

	   return up_sample

	else:
	  
	   x = {}
	   input_i = input
	   for i in range(self.total_block/2): 
               x[i] = self._modules["down_"+str(i)](input_i)
	       input_i = x[i]	

	   # concatenate the hidden state vector from LSTM
	   hidden = {}    
	   # do the filter across different filter and save it as hidden states
	   for i in range(self.filter_bank_size): 
               hidden[i] = self._modules["filter_"+str(i)](input_i)

	   up_sample = {}
	   for  k in range(self.filter_bank_size):
	        x[self.total_block/2] = self._modules["up_"+str(self.total_block/2)](hidden[k])
	        for i in range(self.total_block/2+1, self.total_block):
	            input_i = torch.cat([x[i-1], x[self.total_block-i-1]], 1)	 
	            x[i] = self._modules["up_" + str(i)](input_i)

	        up_sample[k] = x[self.total_block-1]	

	   return up_sample	


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.ELU(inplace=True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ELU(inplace=True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock_Down_Aug(nn.Module):

    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock_Down_Aug, self).__init__()

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.ELU(inplace=True)
        downnorm = nn.BatchNorm2d(inner_nc)

        if outermost:
            model = [downconv]
        elif innermost:
            model = [downrelu, downconv]
        else:
            model = [downrelu, downconv, downnorm]
            
        self.model = nn.Sequential(*model)

    # ignore the wordvec input here
    def forward(self, x):
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock_Up_Aug(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock_Up_Aug, self).__init__()

        uprelu = nn.ELU(inplace=True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            model = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            model = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            model = [uprelu, upconv, upnorm]

        self.model = nn.Sequential(*model)

    # ignore the wordvec input here
    def forward(self, x):
        return self.model(x)
        

# Defines the LSTM Decoder
class EncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, visual_size, num_layers, cell, wordvec):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
	self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        if cell == 'lstm':
           self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
	   self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # since this bi-directional network
        self.linear = nn.Linear(2*hidden_size, visual_size)
        self.wordvec = wordvec
	self.init_weights()
	
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
	# initialize with glove word embedding
	self.embed.weight.data.copy_(torch.from_numpy(self.wordvec))

    # the forward method, which compute the hidden state vector
    def forward(self, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.rnn(packed)

        # only get the last hidden states, should be careful here, only support batch size = 1
	forward = torch.unsqueeze(hiddens[0][-1][self.hidden_size:], 0)
	backward = torch.unsqueeze(hiddens[0][0][0:self.hidden_size],0)
        attvec = F.relu(self.linear(torch.cat((forward, backward), 1)))
        return attvec


# Defines the LSTM Decoder
class EncoderRNN_Linear(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, label_size, num_layers, cell, wordvec):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN_Linear, self).__init__()
	self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        if cell == 'lstm':
           self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
	   self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # since this bi-directional network
        self.linear = nn.Linear(2*hidden_size, label_size)
        self.wordvec = wordvec
	self.init_weights()
	
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
	# initialize with glove word embedding
	self.embed.weight.data.copy_(torch.from_numpy(self.wordvec))

    # forward method to compute the attention over different bucket
    # different from previous lstm, we add the linear layer here   
    def forward(self, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.rnn(packed)

        # only get the last hidden states, should be careful here, only support batch size = 1
	forward = torch.unsqueeze(hiddens[0][-1][self.hidden_size:], 0)
	backward = torch.unsqueeze(hiddens[0][0][0:self.hidden_size],0)
        response = self.linear(torch.cat((forward, backward), 1))
        return response



# encoder using the resnet, it has constrains the input size must be 224*224
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features

# CNN archicture like resnet, but without upsampling, the number of output channel is output_nc
class ResnetCNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, n_blocks, visual_size, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetCNN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

	# currently we fix it, hai wang--5/19/2017
	self.linear = nn.Linear(112896L, visual_size)

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf),
                 nn.ELU(inplace=True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ELU(inplace=True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]

        model += [nn.MaxPool2d(6)]
	self.model = nn.Sequential(*model)

    def forward(self, input):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            x = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
	else:
            x = self.model(input)
	x = x.view(x.size(0), -1)
	return self.linear(x)	


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=2),
            nn.ELU(inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=2),
                # TODO: use InstanceNorm
                nn.BatchNorm2d(ndf * nf_mult),
                nn.ELU(inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=1, stride=2, padding=2),
            # TODO: useInstanceNorm
            nn.BatchNorm2d(ndf * nf_mult),
            nn.ELU(inplace=True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=1)]
        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Instance Normalization layer from
# https://github.com/darkstar112358/fast-neural-style
class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
