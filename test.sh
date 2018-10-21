#!/bin/bash

source ~/.bashrc
lambda_p=0
data="./datasets/all_data_3/"
caption="./datasets/all_data3_pos_neg.json"
checkpoint="all_data_3"

CUDA_VISIBLE_DEVICES=3 python test.py --dataroot $data --caption $caption --name fivek_pix2pix2_perp2_${lambda_p}_${checkpoint} --model pix2pix --which_model_netG unet_128 --which_direction BtoA  --phase val  --align_data
