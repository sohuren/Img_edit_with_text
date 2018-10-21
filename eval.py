import argparse
import os
import cv2
from skimage import io, color
import numpy as np
from skimage import measure


def eval_images(args):

    """Resize the images in 'image_dir' and save into 'output_dir'."""
    
    images = os.listdir(args.image_dir)
    num_images = len(images)

    psnr = {}
    ssim = {}
    lab = {}
    mse = {}
	
    # calculate the image error 
    for i, image in enumerate(images):
       if image.endswith('.png'):
	   if image.find('fake') != -1:
	         
	       image2 = image.replace('fake', 'real')
	       # read two image first	
 	       img1 = io.imread(os.path.join(args.image_dir, image))
 	       img2 = io.imread(os.path.join(args.image_dir, image2))
	       psnr[image2] = measure.compare_psnr(img2, img1)
	       ssim[image2] = measure.compare_ssim(img2, img1, multichannel=True)
	       mse[image2] = measure.compare_mse(img1, img2)		  		
	       # compute the lab error		      
	       lab1 = color.rgb2lab(img1)   
   	       lab2 = color.rgb2lab(img2)
	       area = lab2.shape[1]*lab2.shape[0]	
	       error = np.linalg.norm(abs(lab1[:,:,0] - lab2[:,:,0]))/np.sqrt(area) 	 
	       lab[image2] = error	      	 	     

    # output it to the txt file	     
    fp = open(args.psnr_txt, "wt+")    
    for name in psnr.keys():
        fp.write(name +":" + str(psnr[name]) + "\n")
    fp.write("mean:{},std:{}".format(np.mean(psnr.values()), np.std(psnr.values())))
    fp.close()	

    # output it to the txt file	     
    fp = open(args.ssim_txt, "wt+")	
    for name in ssim.keys():
        fp.write(name +":" + str(ssim[name]) + "\n")
		
    fp.write("mean:{},std:{}".format(np.mean(ssim.values()), np.std(ssim.values())))
    fp.close()	

    # output it to the txt file	     
    fp = open(args.mse_txt, "wt+")	
    for name in mse.keys():
        fp.write(name +":" + str(mse[name]) + "\n")
    fp.write("mean:{},std:{}".format(np.mean(mse.values()), np.std(mse.values())))		
    fp.close()	
	
    # output it to the txt file	     
    fp = open(args.lab_txt, "wt+")	
    for name in lab.keys():
        fp.write(name +":" + str(lab[name]) + "\n")		 	  
    
    fp.write("mean:{},std:{}".format(np.mean(lab.values()), np.std(lab.values())))		
    fp.close()
	
	
def main(args):

    eval_images(args)


# evaluate the images given the generated image
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./enhancement_end2end/train/end2end/images',
                        help='directory for evaluting images')
    # output file	
    parser.add_argument('--psnr_txt', type=str, default='./eval/psnr_train_end.txt',
                        help='psnr txt file')
    parser.add_argument('--ssim_txt', type=str, default='./eval/ssim_train_end.txt',
                        help='ssim txt file')
    parser.add_argument('--mse_txt', type=str, default='./eval/mse_train_end.txt',
                        help='mse txt file')
    parser.add_argument('--lab_txt', type=str, default='./eval/lab_train_end.txt',
                        help='lab txt file')

    args = parser.parse_args()
    main(args)
