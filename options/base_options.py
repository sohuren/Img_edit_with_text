import argparse
import os
from util import util
from pdb import set_trace as st
class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

        self.parser.add_argument('--loadSize', type=int, default=-1, help='scale images to this size, this is the short edge for one image')
        self.parser.add_argument('--fineSize', type=int, default=-1, help='then crop to this size, if -1, then no crop')

        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--flip'  , action='store_true', help='if flip the images for data argumentation')
        self.parser.add_argument('--name', type=str, default='experiment name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--align_data', action='store_true',
                                help='if True, the datasets are loaded from "test" and "train" directories and the data pairs are aligned')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. pix2pix_bucket, pix2pix, ...')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='batch normalization or instance normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        
        self.parser.add_argument('--embed_size', type=int , default=100, help='dimension of word embedding vectors')
        self.parser.add_argument('--hidden_size', type=int , default=128, help='dimension of lstm hidden states')
        self.parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
        self.parser.add_argument('--vocab_path', type=str, default='./datasets/vocab.pkl', help='path for vocabulary wrapper')
        self.parser.add_argument('--rnn', type=str, default='lstm', help='lstm or gru')	
	self.parser.add_argument('--augment_data', type=bool, default=False, help='augment the data')
        self.parser.add_argument('--num_G', type=int, default=4, help='number of buckets we want to use')

        self.parser.add_argument('--visual_size', type=int, default=128, help='embedding size for the visual generated image')
        self.parser.add_argument('--visual_size_dis', type=int, default=64, help='embedding size for the visual generated image (discrinator)')

	self.parser.add_argument('--fusion', type=bool, default=True, help='fusion or argmax when combine with different buckets')
	self.parser.add_argument('--bucket_description', type=str, default='./datasets/bucket_description.json', help='description for each bucket, it is should be a json file')
	self.parser.add_argument('--triple', type=bool, default=True, help='whether we use the triple loss or not')
	self.parser.add_argument('--wordvec', type=str, default='./datasets/vocab_vec_100d.pkl', help='initialize the word embedding with pre-trained glove file')
	self.parser.add_argument('--pre_trained_bucket', type=str, default='./datasets/pre-train_buckets', help='pre-trained buckets')
	self.parser.add_argument('--caption', type=str, default='./datasets/CIM_annotations_200_joint.json', help='path for vocabulary wrapper')

	self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))


        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
