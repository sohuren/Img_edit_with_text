
#!/bin/bash
export PYTHONPATH="/home/t-haiwan/third-party/pytorch-CycleGAN-and-pix2pix-filterbank":$PYTHONPATH

source ~/.bashrc
lambda_p=0
data="./datasets/all_data_3/"
caption="./datasets/all_data3_pos_neg.json"
checkpoint="all_data_3"

CUDA_VISIBLE_DEVICES=2 python train.py --dataroot $data --caption $caption --name fivek_pix2pix2_perp2_${lambda_p}_${checkpoint} --model pix2pix --which_model_netG unet_128 --which_direction BtoA --lambda_A 200 --align_data --no_lsgan --batchSize 1 --nThreads 1 --lambda_p ${lambda_p}
