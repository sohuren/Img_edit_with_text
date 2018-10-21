
#!/bin/bash
export PYTHONPATH="/home/t-haiwan/third-party/pytorch-CycleGAN-and-pix2pix-ImageCaption":$PYTHONPATH
source ~/.bashrc

lambda_p=0
data=all_data
annotation=all_data.json

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/${data} --caption ./datasets/${annotation} --name fivek_pix2pix2_perp2_${lambda_p}_${data}_bucket3 --model pix2pix_bucket3 --which_model_netG unet_128 --which_direction BtoA --lambda_A 200 --align_data --no_lsgan --batchSize 1 --nThreads 1 --lambda_p ${lambda_p}
