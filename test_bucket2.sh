#!/bin/bash

source ~/.bashrc
lambda_p=0

CUDA_VISIBLE_DEVICES=3 python test.py  --dataroot ./datasets/all_data/ --caption ./datasets/all_data.json --name fivek_pix2pix2_perp2_${lambda_p}_all_data_bucket --model pix2pix_bucket2 --which_model_netG unet_128 --which_direction BtoA  --phase test  --align_data --fusion true
