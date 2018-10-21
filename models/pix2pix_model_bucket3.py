import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
from util.build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from PIL import Image
import pickle

# this model contains one lstm for both bucket description and for the user input speech
# every other thing is the same previous model except they we have one lstm in generator  
class Pix2PixModel_Bucket3(BaseModel):
    def name(self):
        return 'Pix2PixModel_Bucket3'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

	self.triple = opt.triple

        # load/define networks
        if opt.lambda_p != 0:
	   self.vgg_model = opt.vgg_model
        else:
	   self.vgg_model = ''

	if opt.wordvec:
           fp = open(opt.wordvec, 'rb')
           self.wordvec = pickle.load(fp)

        self.vocab = opt.vocab
        self.num_G = opt.num_G
        self.netG = {}
        self.netMLP = {}
        self.softmax = torch.nn.Softmax()
	self.fusion = opt.fusion
	self.pre_trained_bucket = opt.pre_trained_bucket

        for i in range(opt.num_G): 

            self.netG[i] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
	                            opt.which_model_netG, opt.norm, self.gpu_ids)
	    # no parameters required
	    for param in self.netG[i].parameters():
                param.requires_grad = False

	    print "load the bucket model"	 
            self.load_network_bucket(self.netG[i], self.pre_trained_bucket + '/bucket' + str(i+1) + '/latest_net_G.pth')		
	    # if we already have the pre-trained G for different models, 
	    # then we can set the require_grad as False	

        # encoder for the image captioning
        self.netEn = networks.define_ENcoder(opt.embed_size, opt.hidden_size, opt.vocab_size, opt.visual_size, opt.num_layers, opt.rnn, self.wordvec, self.gpu_ids)
	
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc + opt.visual_size_dis, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
	    self.netEnDis = networks.define_ENcoder(opt.embed_size, opt.hidden_size, 
					 opt.vocab_size, opt.visual_size_dis, 
					 opt.num_layers, opt.rnn, self.wordvec, self.gpu_ids)	

        if not self.isTrain or opt.continue_train:
	    
            self.load_network(self.netEn, 'En', opt.which_epoch)

            if self.isTrain:
		print "load the discrinator"
                self.load_network(self.netD, 'D', opt.which_epoch)
		self.load_network(self.netEnDis, 'EnDis', opt.which_epoch)

	# all the parameters for the generators
	if self.isTrain:
	    Generator_params = list(self.netEn.parameters())
	    Discrinator_params = list(self.netD.parameters()) + list(self.netEnDis.parameters())	

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss()
				
	    if opt.lambda_p != 0:
	       self.criterionMSE = torch.nn.MSELoss()	

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(Generator_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(Discrinator_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))	

            print('---------- Networks initialized -------------')
            
            networks.print_network(self.netD)
            networks.print_network(self.netEn)
	    networks.print_network(self.netEnDis)
            print('-----------------------------------------------')

    def set_input(self, input):

        AtoB = self.opt.which_direction is 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_description = input['des'] # description

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.description = input_description

	# this is the description for each bucket	
	self.des_bucket = input['des_bucket']
	# this is the des negative one
        self.description_neg = input['des_neg']

	
    def forward(self):        
	# input A
	self.real_A = Variable(self.input_A)
	# user input txt
        descriptions = Variable(self.description.long())
        if torch.cuda.is_available():
            descriptions = descriptions.cuda()
	length = [descriptions.size()[1]]  
        word_vec = self.netEn.forward(descriptions, length)

	# this is the one used for descriptions for each bucket  	
        # now get the attetion weight for each different models	
        self.fake_B_bucket = {}
        self.fake_B_attention = {}

        for i in range(self.num_G):
            self.fake_B_bucket[i] = self.netG[i].forward(self.real_A)
	    # only checking the first description now 
	    # we can extend it to multiple later
	    description_bucket = Variable(self.des_bucket[str(i)][0].long())
	    if torch.cuda.is_available(): 
               description_bucket = description_bucket.cuda()	 
	    length = [description_bucket.size(1)]	   		
            visual_vec = self.netEn.forward(description_bucket, length)  #take the text associated with the bucket
            self.fake_B_attention[i] = word_vec.dot(visual_vec)[None]
		
        attention = self.fake_B_attention[0]
        for i in range(1, self.num_G):
            attention = torch.cat((attention, self.fake_B_attention[i]), 1)    
	attention = self.softmax(attention)
	atts = torch.split(attention, 1, 1)
		
	img_size = self.fake_B_bucket[0].size()
	self.fake_B = self.fake_B_bucket[0]*atts[0].expand(img_size)
	for i in range(1, self.num_G):  
            self.fake_B = self.fake_B + self.fake_B_bucket[i]*atts[i].expand(img_size)     
	
        self.real_B = Variable(self.input_B)
	
    # no backprop gradients
    def test(self):

        self.real_A = Variable(self.input_A, volatile=True) 
        descriptions = Variable(self.description.long(), volatile=True)
        if torch.cuda.is_available():
            descriptions = descriptions.cuda()
        length = [descriptions.size()[1]]  
        word_vec = self.netEn.forward(descriptions, length)
		
        # now get the attetion weight for each different models	
        self.fake_B_bucket = {}
        self.fake_B_attention = {}
        for i in range(self.num_G):

            self.fake_B_bucket[i] = self.netG[i].forward(self.real_A)
            description_bucket = Variable(self.des_bucket[str(i)][0].long(), volatile=True)
            if torch.cuda.is_available():
               description_bucket = description_bucket.cuda()
            length = [description_bucket.size(1)]   		     	
            visual_vec = self.netEn.forward(description_bucket, length)
            self.fake_B_attention[i] = word_vec.dot(visual_vec)[None]

        attention = self.fake_B_attention[0]
        for i in range(1, self.num_G):
            attention = torch.cat((attention, self.fake_B_attention[i]), 1)    
	attention = self.softmax(attention)

	print attention[0].data

	# fusion with different weights or just take the argmax
	if self.fusion:
	   atts = torch.split(attention, 1, 1)
	   img_size = self.fake_B_bucket[0].size()
	   self.fake_B = self.fake_B_bucket[0]*atts[0].expand(img_size)
	   for i in range(1, self.num_G):  
               self.fake_B = self.fake_B + self.fake_B_bucket[i]*atts[i].expand(img_size)
	else: 
	   values, indices = torch.max(attention, 1) 
           self.fake_B = self.fake_B_bucket[indices]

        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):

	# calculate the lstm in the discrinator
	if not self.triple:	
           descriptions = Variable(self.description.long())
           if torch.cuda.is_available():
              descriptions = descriptions.cuda()
           length = [descriptions.size()[1]]
           word_vec = self.netEnDis.forward(descriptions, length)

           # stop backprop to the generator by detaching fake_B	
    	   fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))	
           fake_AB_shape = fake_AB.detach().size()
           word_vec = torch.unsqueeze(word_vec, 2)
           word_vec = torch.unsqueeze(word_vec, 3)
           word_vec = word_vec.expand(fake_AB_shape[0], word_vec.size(1), fake_AB_shape[2], fake_AB_shape[3])
           fake_AB_discription = torch.cat([fake_AB.detach(), word_vec], 1)
           self.pred_fake = self.netD.forward(fake_AB_discription)
           self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

           # Real
	   real_AB = torch.cat((self.real_A, self.real_B), 1)
           real_AB_discription = torch.cat([real_AB, word_vec], 1)
           self.pred_real = self.netD.forward(real_AB_discription)
           self.loss_D_real = self.criterionGAN(self.pred_real, True)
	
           # Combined loss
           self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
           self.loss_D.backward()

	else:

	   # real text
           descriptions = Variable(self.description.long())
           if torch.cuda.is_available():
              descriptions = descriptions.cuda()
           length = [descriptions.size()[1]]
           word_vec = self.netEnDis.forward(descriptions, length)
           # stop backprop to the generator by detaching fake_B
           fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
           fake_AB_shape = fake_AB.detach().size()
           word_vec = torch.unsqueeze(word_vec, 2)
           word_vec = torch.unsqueeze(word_vec, 3)
           word_vec = word_vec.expand(fake_AB_shape[0], word_vec.size(1), fake_AB_shape[2], fake_AB_shape[3])

           # fake text
           descriptions_neg = Variable(self.description_neg.long())
           if torch.cuda.is_available():
              descriptions_neg = descriptions_neg.cuda()
           length = [descriptions_neg.size()[1]]
           word_vec_neg = self.netEnDis.forward(descriptions_neg, length)
           # stop backprop to the generator by detaching fake_B
           word_vec_neg = torch.unsqueeze(word_vec_neg, 2)
           word_vec_neg = torch.unsqueeze(word_vec_neg, 3)
           word_vec_neg = word_vec_neg.expand(fake_AB_shape[0], word_vec_neg.size(1), fake_AB_shape[2], fake_AB_shape[3])

           # real text + fake image pair
           fake_AB_discription = torch.cat([fake_AB.detach(), word_vec], 1)
           self.pred_fake = self.netD.forward(fake_AB_discription)
           self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

           # real text + real image pair
           real_AB = torch.cat((self.real_A, self.real_B), 1)
           real_AB_discription = torch.cat([real_AB, word_vec], 1)
           self.pred_real = self.netD.forward(real_AB_discription)
           self.loss_D_real = self.criterionGAN(self.pred_real, True)

           # fake text + real image pair
           real_AB_discription = torch.cat([real_AB, word_vec_neg], 1)
           self.pred_real = self.netD.forward(real_AB_discription)
           self.loss_D_fake += self.criterionGAN(self.pred_real, False)

           # Combined loss with average weight
           self.loss_D = self.loss_D_fake*0.333 + self.loss_D_real*0.333
           self.loss_D.backward()  		

    def backward_G(self):

        # First, G(A) should fake the discriminator
        descriptions = Variable(self.description.long())
        if torch.cuda.is_available():
            descriptions = descriptions.cuda()
        length = [descriptions.size()[1]]
        word_vec = self.netEnDis.forward(descriptions, length)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_AB_shape = fake_AB.size()
        word_vec = torch.unsqueeze(word_vec, 2)
        word_vec = torch.unsqueeze(word_vec, 3)
        word_vec = word_vec.expand(fake_AB_shape[0], word_vec.size(1), fake_AB_shape[2], fake_AB_shape[3])
        fake_AB_discription = torch.cat((self.real_A, self.fake_B, word_vec), 1)

        pred_fake = self.netD.forward(fake_AB_discription)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

	# Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

	# Third, Content loss, the fake_B, real_B in ranges [0,1]
	if self.vgg_model:
	   feature_fake = self.vgg_model(self.fake_B)
	   feature_real = self.vgg_model(self.real_B)
	   # depends on which layer we want to use
	   feature_relu2_2 = Variable(feature_real[1].data, requires_grad=False)
	   self.loss_G_Perp = self.criterionMSE(feature_fake[1], feature_relu2_2)
	else:
	   self.loss_G_Perp = Variable(torch.FloatTensor([0]))	

	self.real_B.cuda()
	self.fake_B.cuda()
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

	if self.vgg_model:
	   self.loss_G += self.loss_G_Perp*self.opt.lambda_p

	self.loss_G.backward(retain_variables=True)	

    def optimize_parameters(self):

        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

	self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                	('G_L1', self.loss_G_L1.data[0]),
		        ('G_Perp', self.loss_G_Perp.data[0]),
                	('D_real', self.loss_D_real.data[0]),
                	('D_fake', self.loss_D_fake.data[0]),
		        ('G', self.loss_G.data[0]),
		        ('D', self.loss_D.data[0])
        ])

    def get_current_visuals(self):

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
	# the corresponding text
        text = util.idx2text(self.description, self.vocab)	
        return OrderedDict([('real_A:'+text, real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):

        use_gpu = self.gpu_ids is not None
	self.save_network(self.netD, 'D', label, use_gpu)
	self.save_network(self.netEn, 'En', label, use_gpu)
	self.save_network(self.netEnDis, 'EnDis', label, use_gpu)

    def update_learning_rate(self):

        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
	
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
