
#!/bin/bash
export PYTHONPATH="/home/t-haiwan/third-party/pytorch-CycleGAN-and-pix2pix-ImageCaption":$PYTHONPATH
source ~/.bashrc

lambda_p=0
data="./datasets/all_data_3"
annotation="./datasets/all_data3_pos_neg.json"
checkpoint="all_data3_model_bucket4"

CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ${data} --caption ${annotation} --name fivek_pix2pix2_perp2_${lambda_p}_${checkpoint} --model pix2pix_bucket5 --which_model_netG unet_128 --which_direction BtoA --lambda_A 200 --align_data --no_lsgan --batchSize 1 --nThreads 1 --lambda_p ${lambda_p} --num_G 5
