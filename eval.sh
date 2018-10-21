python eval.py  --image_dir ./enhancement_end2end/train/end2end/images --psnr_txt ./eval/psnr_train_end.txt --ssim_txt ./eval/ssim_train_end.txt --mse_txt ./eval/mse_train_end.txt --lab_txt ./eval/lab_train_end.txt
python eval.py  --image_dir ./enhancement_end2end/train/WTA/images --psnr_txt ./eval/psnr_train_WTA.txt --ssim_txt ./eval/ssim_train_WTA.txt --mse_txt ./eval/mse_train_WTA.txt --lab_txt ./eval/lab_train_WTA.txt
python eval.py  --image_dir ./enhancement_end2end/train/AVE/images --psnr_txt ./eval/psnr_train_AVE.txt --ssim_txt ./eval/ssim_train_AVE.txt --mse_txt ./eval/mse_train_AVE.txt --lab_txt ./eval/lab_train_AVE.txt

python eval.py  --image_dir ./enhancement_end2end/val/end2end/images --psnr_txt ./eval/psnr_val_end.txt --ssim_txt ./eval/ssim_val_end.txt --mse_txt ./eval/mse_val_end.txt --lab_txt ./eval/lab_val_end.txt
python eval.py  --image_dir ./enhancement_end2end/val/WTA/images --psnr_txt ./eval/psnr_val_WTA.txt --ssim_txt ./eval/ssim_val_WTA.txt --mse_txt ./eval/mse_val_WTA.txt --lab_txt ./eval/lab_val_WTA.txt
python eval.py  --image_dir ./enhancement_end2end/val/AVE/images --psnr_txt ./eval/psnr_val_AVE.txt --ssim_txt ./eval/ssim_val_AVE.txt --mse_txt ./eval/mse_val_AVE.txt --lab_txt ./eval/lab_val_AVE.txt
