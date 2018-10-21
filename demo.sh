#!/bin/bash
source ~/.bashrc
lambda_p=0

CUDA_VISIBLE_DEVICES=3 python demo.py --name fivek_pix2pix2_perp2_${lambda_p}_all_data_3 --model pix2pix --which_model_netG unet_128  --image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A.png" --description "i want to use this image" --saved_image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A_keep_the_same.png" --align_data

CUDA_VISIBLE_DEVICES=3 python demo.py --name fivek_pix2pix2_perp2_${lambda_p}_all_data_3 --model pix2pix --which_model_netG unet_128  --image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A.png" --description "image is so good" --saved_image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A_imae_is_good.png" --align_data

CUDA_VISIBLE_DEVICES=3 python demo.py --name fivek_pix2pix2_perp2_${lambda_p}_all_data_3 --model pix2pix --which_model_netG unet_128  --image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A.png" --description "increase brightness" --saved_image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A_increase_brightness.png" --align_data

CUDA_VISIBLE_DEVICES=3 python demo.py --name fivek_pix2pix2_perp2_${lambda_p}_all_data_3 --model pix2pix --which_model_netG unet_128  --image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A.png" --description "increase contrast" --saved_image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A_increase_contrast.png" --align_data


CUDA_VISIBLE_DEVICES=3 python demo.py --name fivek_pix2pix2_perp2_${lambda_p}_all_data_3 --model pix2pix --which_model_netG unet_128  --image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A.png" --description "reduce darkness" --saved_image "./results/fivek_pix2pix2_perp2_0_all_data_3/test_latest/images/expertA-expertA-a4030-dvf_092_real_A_reduce_darkness.png" --align_data
