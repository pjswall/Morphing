
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='EVAL_UNMIX')
parser.add_argument('--dataset', type=str, default='dogs', metavar='str',
                    help='dataset name from [dogs, imagenetsubset],'
                         '(default: dogs)')
parser.add_argument('--test_mode', type=str, default='Morph', 
                    help='select test type for the model, from [Morph, Nonmorph]')
parser.add_argument('--in_size', type=int, default=256, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='./eval_output', metavar='str',
                    help='evaluation output dir (default: ./eval_output)')
args = parser.parse_args()



def cpt_psnr_ssim(G_pred1, G_pred2, gt1, gt2):
    psnr1 = 0.5*utils.cpt_rgb_psnr(G_pred1, gt1, PIXEL_MAX=1.0) + \
            0.5*utils.cpt_rgb_psnr(G_pred2, gt2, PIXEL_MAX=1.0)
    psnr2 = 0.5*utils.cpt_rgb_psnr(G_pred1, gt2, PIXEL_MAX=1.0) + \
            0.5*utils.cpt_rgb_psnr(G_pred2, gt1, PIXEL_MAX=1.0)

    ssim1 = 0.5*utils.cpt_rgb_ssim(G_pred1, gt1) + \
            0.5*utils.cpt_rgb_ssim(G_pred2, gt2)
    ssim2 = 0.5*utils.cpt_rgb_ssim(G_pred1, gt2) + \
            0.5*utils.cpt_rgb_ssim(G_pred2, gt1)

    return max(psnr1, psnr2), max(ssim1, ssim2)



def load_model(args):

    net_G = cycnet.define_G(
                input_nc=3, output_nc=6, ngf=64, netG=args.net_G, use_dropout=False, norm='none').to(device)
    print('loading the best checkpoint...')
    checkpoint = torch.load(os.path.join(args.ckptdir, 'best_ckpt.pt'))
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return net_G



def run_eval(args):

    val_dirs = []
    val1_dirs =[]

    current_path = os.getcwd()
    orig_titles = os.listdir(r'./datasets/AMSL/orig/')
    morph_titles = os.listdir(r'./datasets/AMSL/Morph/val/')
    non_morph_titles = os.listdir(r'./datasets/AMSL/NonMorph/')

    orig_img_map = dict()
    morph_map = dict()
    non_morph_map = dict()

    morgan_morph_titles= os.listdir(r'./datasets/MOR/eval_test/')
    
    print('running evaluation...')

    if args.save_output:
        if os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

    running_psnr = []
    running_ssim = []
        
    if args.dataset == 'AMSL':
        
        for title in orig_titles:
            orig_img_map[title[:3]] = title
                
        for title in morph_titles:
            morph_map[title[:10]] = title
            
        for title in non_morph_titles:
            non_morph_map[title[:6]] = title
            

        for title in morph_titles:      
            title = title[:10]
            if title[:3] == '.ip':
                pass
            else:
                val_dirs.append(current_path +"/datasets/AMSL/Morph/val/"+morph_map[title[:10]])
                
        for title in non_morph_titles:
            val1_dirs.append(current_path+"/datasets/AMSL/NonMorph/"+non_morph_map[title[:6]])

    elif args.dataset == 'MOR' or args.dataset == 'EMOR' or args.dataset == 'REGEN':

        pattern = re.compile(r'\d+\_\d+\_\d+')
        for title in morph_titles:

            m = re.search(pattern,title)

            if m is not None:

                m1 = m.group(0)
                self.dir3.append(self.current_path +"/datasets/MOR/Morphed_Test/"+"Mor"+m1+".jpg")
   
    for idx in range(len(val1_dirs)):

        if args.dataset == 'AMSL':

            if args.test_mode == 'Morph':
            
                str = val_dirs[idx]
                path1 = current_path +"/datasets/AMSL/orig/"+ orig_img_map[str[-14:-11]]
                path2 = current_path +"/datasets/AMSL/orig/"+ orig_img_map[str[-7:-4]] 
                path3 = str

                img1 = cv2.imread(path1,cv2.IMREAD_COLOR)
                img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
                img2 = cv2.imread(path2,cv2.IMREAD_COLOR)       
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                img3 = cv2.imread(path3,cv2.IMREAD_COLOR)
                img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)


            else:
            
                path4 = val1_dirs[idx]                           #For non morph images
                
                img1 = cv2.imread(path4,cv2.IMREAD_COLOR)
                img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
                img2 = cv2.imread(path4,cv2.IMREAD_COLOR)        
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                img3 = cv2.imread(path4,cv2.IMREAD_COLOR)
                img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

        elif args.dataset == 'MOR' or args.dataset == 'EMOR' or args.dataset == 'REGEN':
            
            if args.test_mode == 'Morph':

                str = val_dirs[idx]

                pattern = re.compile(r'\d+\_\d+\_\d+')
                m = re.search(pattern,str)

                m1 = m.group(0)

                path1 = self.current_path + "/datasets/MOR/set1/"+ "Mor" +m1 +".jpg"
                path2 = self.current_path + "/datasets/MOR/set1/"+ "Mor" +m1 +".jpg"
                path3 = str

                img1 = cv2.imread(path1,cv2.IMREAD_COLOR)
                img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
                img2 = cv2.imread(path2,cv2.IMREAD_COLOR)       # For Morph
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                img3 = cv2.imread(path3,cv2.IMREAD_COLOR)
                img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

            else:
            
                path4 = val1_dirs[idx]                           #For non morph images
                
                img1 = cv2.imread(path4,cv2.IMREAD_COLOR)
                img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
                img2 = cv2.imread(path4,cv2.IMREAD_COLOR)        
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                img3 = cv2.imread(path4,cv2.IMREAD_COLOR)
                img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)


        gt1 = TF.resize(TF.to_pil_image(img1), [args.in_size, args.in_size])
        gt1 = TF.to_tensor(gt1).unsqueeze(0)
        gt2 = TF.resize(TF.to_pil_image(img2), [args.in_size, args.in_size])
        gt2 = TF.to_tensor(gt2).unsqueeze(0)
        img_mix = TF.resize(TF.to_pil_image(img3), [args.in_size, args.in_size])
        img_mix = TF.to_tensor(img_mix).unsqueeze(0)

        with torch.no_grad():
            out = net_G(img_mix.to(device))
            G_pred1 = out[:, 0:3, :, :]
            G_pred2 = out[:, 3:6, :, :]

        G_pred1 = np.array(G_pred1.cpu().detach())
        G_pred1 = G_pred1[0, :].transpose([1, 2, 0])
        G_pred2 = np.array(G_pred2.cpu().detach())
        G_pred2 = G_pred2[0, :].transpose([1, 2, 0])
        gt1 = np.array(gt1.cpu().detach())
        gt1 = gt1[0, :].transpose([1, 2, 0])
        gt2 = np.array(gt2.cpu().detach())
        gt2 = gt2[0, :].transpose([1, 2, 0])
        img_mix = np.array(img_mix.cpu().detach())
        img_mix = img_mix[0, :].transpose([1, 2, 0])

        # G_pred1[G_pred1 > 0.5] = 0.5
        # G_pred1[G_pred1 < 0] = 0
        # G_pred2[G_pred2 > 0.5] = 0.5
        # G_pred2[G_pred2 < 0] = 0

        psnr, ssim = cpt_psnr_ssim(G_pred1, G_pred2, gt1, gt2)
        running_psnr.append(psnr)
        running_ssim.append(ssim)

        if args.save_output:
              
                plt.imsave(os.path.join("./eval_output/"+args.dataset+"/"+args.test_mode+path4[-14:-4] + '_input.png'), img_mix)
                plt.imsave(os.path.join("./eval_output/"+args.dataset+"/"+args.test_mode+path4[-14:-4] + '_gt1.png'), gt1)
                plt.imsave(os.path.join("./eval_output/"+args.dataset+"/"+args.test_mode+path4[-14:-4] + '_gt2.png'), gt2)
                plt.imsave(os.path.join("./eval_output/"+args.dataset+"/"+args.test_mode+path4[-14:-4] + '_output1.png'), G_pred1)
                plt.imsave(os.path.join("./eval_output/"+args.dataset+"/"+args.test_mode+path4[-14:-4] + '_output2.png'), G_pred2)

        print('id: %d, running psnr: %.4f, running ssim: %.4f'
              % (idx, np.mean(running_psnr), np.mean(running_ssim)))

    print('Dataset: %s, average psnr: %.4f, average ssim: %.4f'
          % (args.dataset, np.mean(running_psnr), np.mean(running_ssim)))



if __name__ == '__main__':

    args.save_output = True

    net_G = load_model(args)
    run_eval(args)




