import time
import torch
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.Vgg16 import Vgg16Part
from util.util import init_vgg16
import pickle
from util.build_vocab import Vocabulary

# Load vocabulary wrapper.
with open(opt.vocab_path, 'rb') as f:
     vocab = pickle.load(f)

opt.vocab = vocab
opt.vocab_size = len(vocab)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
num_train = len(data_loader)
print('#training images = %d' % num_train)


print('load vgg16 models')
init_vgg16("vgg_model")
vgg_model = Vgg16Part()
vgg_model.load_state_dict(torch.load('vgg_model/vgg16.weight'))
if torch.cuda.is_available():	
   vgg_model.cuda()
opt.vgg_model = vgg_model


model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps % num_train

	
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, epoch_iter, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
