
# deep learning model in PyTorch, the model takes the txt,image as input and output a new edited image

## Prerequisites
- Linux.
- Python 2.7.
- CPU or NVIDIA GPU + CUDA CuDNN.
- corresponding pytorch version (tested on the newest version)

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/
- Install python libraries [dominate](https://github.com/Knio/dominate).
- Install other necessary libraries such as Pillow, sklearn etc


- Train a end to end model:
bash train.sh
To view results as the model trains, check out the html file `./checkpoints/XXX/web/index.html` where XXX is specified by your train.sh shell script
- Test the model:
bash test.sh
The val results will be saved to a html file here: `./results/XXX/latest_val/index.html`.


- Train different buckets model, first please pre-train the each buckets on other folders and then copy all the folders here (train_bucket2.sh is also fine, but no not elegant, train_bucket.sh is not used now):
bash train_bucket3.sh or train_bucket4.sh
To view results as the model trains, check out the html file `./checkpoints/XXX/web/index.html` where XXX is specified by your train.sh shell script
- Test the model:
bash test.sh
The test results will be saved to a html file here: `./results/XXX/latest_val/index.html`.

- To see the demo, i.e., given the txt and image and trained model, generate a new image and save it, run:
bash demo.sh
you can specify the model, trained model location, input image, text operation and path of output image


## Training/test Details
- See `options/train_options.py` and `options/base_options.py` for training flags; 
- see `optoins/test_options.py` and `options/base_options.py` for test flags.
- CPU/GPU: Set `--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode.
- During training, you can visualize the result of current training. If you set `--display_id 0`, we will periodically save the training results to `[opt.checkpoints_dir]/[opt.name]/web/`. If you set `--display_id` > 0, the results will be shown on a local graphics web server launched by [szym/display: a lightweight display server for Torch](https://github.com/szym/display). To do this, you should have Torch, Python 3, and the display package installed. You need to invoke `th -ldisplay.start 8000 0.0.0.0` to start the server.
- because we use certain layer con, thus is has some requirements on image size, for example, if you use unet_128, then the width&height must can be divided by 128.
- to train all those different models, you must specify the input data dir and input text description, please see the script for details   
- To train a model on your own datasets, you need to create a data folder which contain two images from domain A and B, and then the learn can either learning to map A->B or B->A depends on which direction you want to learn. You can test your model on your training set by setting ``phase='train'`` in  `test.sh`. You can also create subdirectories `testA` and `testB` if you have test data.
Corresponding images in a pair {A,B} must be the same size and have the same filename.
 to create your own such kind of dataset, please see the code in datasets folder, lots of python files exist there    
- you can see the description.json for formatting on the description part, if you want to use own description data, you should format your data as this way and run the util/build_vocab.py to re-build the vocab and re-build the glove embedding file. 
- currently we use glove 100-d to initialize the word embedding matrix and vgg_model to compute the perceputal loss if necessary
- to evaluate different models, put all the ground truth, predictions under the same folder and run eval.py, they will calculate the ssim, psnr etc

## Acknowledgments
Code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
