import argparse
import ray
import ray
from ray import tune
from functools import partial
from argparse import ArgumentParser
import json
import os
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import time
import torchvision.transforms as trns
from torch.optim import Adam, AdamW
from torch.autograd import Variable
import ray
from ray import tune
import wandb
import random
import numpy as np
from PIL import Image

from utils.data_set import InferenceDataset
from utils.model.backbone.backbone import UNetPP, SwinUNet
from utils.utils import adjust_learning_rate, save_images_per_epoch, save_images_per_epoch_conversion, record_parser
from utils.loss import BoundaryGradient, AdaptiveMSSSIM, GradientRegularizer

import nibabel as nib

import matplotlib.pyplot as plt
# def preprocessing():
    # write NiftiDataset(anatomical_nifti_path, functional_nifti_path, mask_nifti_path)
        # 1. create anatomical input from FreeSurfer aparc+aseg.nii and SPM GM/WM/CSF mask
        # 2. rescale functional imaging from SUV/SUVr (store the linear transformation paras(a, b) for post-processing)
        # 3. 3d to 2d slices
    # pass NiftiDataset into Dataloader
    
# def postprocessing():
    # 1. stack 2d back to 3d
    # 2. rescale functional imaging back to SUV/SUVr (use the linear transformation paras(a, b))

def _z_score_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """

    mean = torch.mean(image, dim=(0, 2, 3), keepdim=True)
    std = torch.std(image, dim=(0, 2, 3), keepdim=True)
    return (image - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero

def _reverse_min_max_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """
    
    # non_zero_mask = (image != 0)  # Create a mask where non-zero pixels are True
    # masked_image = torch.where(non_zero_mask, image, torch.tensor(float('inf')))
    # print(image.shape)
    min_val = image.amin(dim=(0, 2, 3), keepdim=True)#torch.min(image, dim=(1, 2), keepdim=True)
    max_val = image.amax(dim=(0, 2, 3), keepdim=True)#torch.max(image, dim=(1, 2), keepdim=True)
    # print("pre: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    image = (image - min_val) / (max_val - min_val + 1e-8)
    # print("post: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    image = torch.ones(image.shape) - image
    # print("reverse: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    return image

def logistic_transform(x, a=5):
    return torch.pow(x, a) / (torch.pow(x, a) + torch.pow(1 - x, a))

def _min_max_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """
    
    # non_zero_mask = (image != 0)  # Create a mask where non-zero pixels are True
    # masked_image = torch.where(non_zero_mask, image, torch.tensor(float('inf')))
    # print(image.shape)
    min_val = image.amin(dim=(0, 2, 3), keepdim=True)#torch.min(image, dim=(1, 2), keepdim=True)
    max_val = image.amax(dim=(0, 2, 3), keepdim=True)#torch.max(image, dim=(1, 2), keepdim=True)
    # print("pre: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    image = (image - min_val) / (max_val - min_val + 1e-8)
    # print("post: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    # print("reverse: ", image.amin(dim=(2, 3), keepdim=False), image.amax(dim=(2, 3), keepdim=False))

    return image

def _data_postprocessing(data, fusion_outputs, anatomical_inputs):
    
    filename = data[3][0]
    # print(np.asarray(data[4]))
    pet_min = np.asarray(data[4][0][0])
    pet_max = np.asarray(data[4][0][1])
    # print(pet_min, pet_max)
    # mr_min=data[4][0][2]
    # mr_max=data[4][0][3]
    # print(fusion_outputs.shape, pet_min.shape)
    # print("pre", np.max(fusion_outputs), np.min(fusion_outputs))
    fusion_outputs = fusion_outputs[:, 0] * (pet_max - pet_min + 1e-5) + pet_min
    # fusion_outputs = fusion_outputs[:, 0]
    # print("post", np.max(fusion_outputs), np.min(fusion_outputs))
    

    fusion_outputs = np.swapaxes(fusion_outputs, 1, 2)
    fusion_outputs = np.swapaxes(fusion_outputs, 0, 2)

    anatomical_inputs=anatomical_inputs[:, 0]
    anatomical_inputs = np.swapaxes(anatomical_inputs, 0, 2)
    anatomical_inputs = np.swapaxes(anatomical_inputs, 0, 1)
    # print(anatomical_inputs.shape)
    if args.suvr:
        if os.path.isfile(os.path.join(args.input_directory, filename, "SUVr.nii")):
            fusion_nifti=nib.Nifti1Image(fusion_outputs[:, :, :, None], affine=nib.load(os.path.join(args.input_directory, filename, "SUVr.nii")).affine)
            file="SUVr"
        else:
            fusion_nifti=nib.Nifti1Image(fusion_outputs[:, :, :, None], affine=nib.load(os.path.join(args.input_directory, filename, "PET.nii")).affine)
            file="PET"
        
    else:
        if os.path.isfile(os.path.join(args.input_directory, filename, "SUV.nii")):
            fusion_nifti=nib.Nifti1Image(fusion_outputs, affine=nib.load(os.path.join(args.input_directory, filename, "SUV.nii")).affine)
            file="SUV"
        else:
            fusion_nifti=nib.Nifti1Image(fusion_outputs[:, :, :, None], affine=nib.load(os.path.join(args.input_directory, filename, "PET.nii")).affine)
            file="PET"
    anatomical_nifti=nib.Nifti1Image(anatomical_inputs, affine=nib.load(os.path.join(args.input_directory, filename, "MR.nii")).affine)
    nib.save(anatomical_nifti, os.path.join(args.input_directory, filename, "anat_{}.nii".format(args.segmentation)))

    if args.segmentation==0:
        nib.save(fusion_nifti, os.path.join(args.input_directory, filename, args.resume_fusion_checkpoint.split('/')[-1][:-4]+"_input{}_full_mr_".format(str(args.simulation))+file+".nii"))
    if args.segmentation==1:
        nib.save(fusion_nifti, os.path.join(args.input_directory, filename, args.resume_fusion_checkpoint.split('/')[-1][:-4]+"_input{}_pseudo_seg_".format(str(args.simulation))+file+".nii"))
        # nib.save(anatomical_nifti, os.path.join(args.input_directory, filename, "pseudo_seg.nii"))
    if args.segmentation==2:
        nib.save(fusion_nifti, os.path.join(args.input_directory, filename, args.resume_fusion_checkpoint.split('/')[-1][:-4]+"_input{}_fs_seg_".format(str(args.simulation))+file+".nii"))
        # nib.save(anatomical_nifti, os.path.join(args.input_directory, filename, "spm_seg.nii"))
    if args.segmentation==3:
        nib.save(fusion_nifti, os.path.join(args.input_directory, filename, args.resume_fusion_checkpoint.split('/')[-1][:-4]+"_input{}_spm_seg_".format(str(args.simulation))+file+".nii"))
        # nib.save(anatomical_nifti, os.path.join(args.input_directory, filename, "fs_seg_.nii"))
        
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 3 , 1)
    # plt.imshow(anatomical_inputs[50, :, :])
    # plt.colorbar()
    # plt.axis("off")
    # plt.subplot(1, 3, 2)
    # plt.imshow(anatomical_inputs[:, 50, :])
    # plt.colorbar()
    # plt.axis("off")
    # plt.subplot(1, 3, 3)
    # plt.imshow(anatomical_inputs[:, :, 50])
    # plt.colorbar()
    # plt.axis("off")
    # plt.show()
    # plt.savefig(os.path.join(args.input_directory, filename, "experiment_seg.png"))

    # plt.figure(figsize=(6, 4))
    # plt.hist(anatomical_inputs[anatomical_inputs!=0].ravel(), bins=50, color='blue', alpha=1)
    # plt.title('Histogram of Image Intensities')
    # plt.xlabel('Intensity Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()
    # plt.savefig(os.path.join(args.input_directory, filename, "histogram.png"))

    return fusion_outputs
    
    
def _data_preprocessing(data, segmentation, cuda=True):
    

    # print(data[0][0].shape)
    # print(data[1][0].shape)
    # print(data[2][0].shape)
    # print(data[3][0])
    # print(data[4][0].shape)
    # print(data[5][0].shape)

        
        

    batch_pet=data[0][0]
    batch_mri=data[1][0]
    batch_label=data[2][0]
    batch_filname=data[3][0]
    batch_mask=data[5][0]
    batch_spm_anatomical=data[6][0]

    #===============================================
    # PET input
    #===============================================
    PET_input = Variable(batch_pet,requires_grad=False)
    
    #===============================================
    # Full T1-MRI
    #===============================================
    MRI_full = Variable(batch_mri, requires_grad=False)
    
    #===============================================
    # Anatomical input
    # 0 = full MRI
    # 1 = Pseudo seg with MR intensity
    # 2 = predefined 4:1:0 SPM
    # 3 = predefined 4:1:0 (FreeSurfer)
    #===============================================
    if segmentation==0:
        anatomical_input = Variable(batch_mri, requires_grad=False)
    elif segmentation==1:

        # mask=torch.where(batch_mri<0.3, torch.tensor(0.0), torch.tensor(1.0))
        # anatomical_input=batch_mri*mask
        # anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        # anatomical_input=anatomical_input*mask
        # anatomical_input = Variable(anatomical_input, requires_grad=False)


        mask=torch.where(batch_mri<0.3, torch.tensor(0.0), torch.tensor(1.0))
        anatomical_input=batch_mri*mask
        anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        anatomical_input=anatomical_input*mask
        anatomical_input=_min_max_normalize(anatomical_input)  
        anatomical_input=logistic_transform(anatomical_input, a=1.5) 
        anatomical_input = Variable(anatomical_input, requires_grad=False)


    elif segmentation==2:
        
        # gm_mask = np.where(np.isin(batch_label.astype(int), [3, 42]), 1, 0)
        # wm_mask = np.where(np.isin(batch_label.astype(int), [2, 41]), 1, 0)
    
        # seg=torch.zeros(MRI_full.shape)
        # seg=np.where(gm_mask[:, :, :, 0] ==1, 1, 0)
        # anatomical_input=np.where(wm_mask[:, :, :, 1] ==1, 0.25, seg)
        gm_mask = torch.isin(batch_label.int(), torch.tensor([1, 3, 6, 27, 29, 30, 40, 42, 45, 59, 61, 62, 80, 81, 82, 7, 46])).int()#[3, 42]
        wm_mask = torch.isin(batch_label.int(), torch.tensor([2, 41, 77, 78, 79, 192, 250, 251, 252, 253, 254, 255, 8, 47])).int()#[2, 41]

        # "Gray Matter": [1, 3, 6, 27, 29, 30, 40, 42, 45, 59, 61, 62, 80, 81, 82, 7, 46],
        # "White Matter": [2, 41, 77, 78, 79, 192, 250, 251, 252, 253, 254, 255, 8, 47],
        # print(torch.max(gm_mask), torch.min(gm_mask))
        # Initialize the segmentation tensor
        seg = torch.zeros_like(MRI_full)

        # Apply GM mask to segmentation tensor
        seg = torch.where(gm_mask == 1, torch.tensor(1.0), seg)

        # Apply WM mask to anatomical input
        seg = torch.where(wm_mask == 1, torch.tensor(0.25), seg)
        anatomical_input = Variable(seg, requires_grad=False)
        # anatomical_input = Variable(batch_mri, requires_grad=False)
    elif segmentation==3:
        
        # TODO <-- SPM

        # mask=torch.where(batch_mri<0.3, torch.tensor(0.0), torch.tensor(1.0))
        # anatomical_input=batch_mri*mask
        # anatomical_input=logistic_transform(anatomical_input, a=1.5) 

        # anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        # anatomical_input=anatomical_input*mask
        # anatomical_input=_min_max_normalize(anatomical_input)  
        # # anatomical_input=logistic_transform(anatomical_input, a=1.5) 
        # anatomical_input = Variable(anatomical_input, requires_grad=False)

        # mask=torch.where(batch_mri<0.3, torch.tensor(0.0), torch.tensor(1.0))
        # anatomical_input=batch_mri*mask
        # anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        # anatomical_input=anatomical_input*mask
        # anatomical_input=_min_max_normalize(anatomical_input)  
        # anatomical_input=logistic_transform(anatomical_input, a=1.5) 
        # anatomical_input = Variable(anatomical_input, requires_grad=False)

        
        anatomical_input = Variable(batch_spm_anatomical, requires_grad=False)

    #===============================================
    # Weighted mask
    #===============================================
    # weighted_mask = Variable(batch_mri[:,3,:,:], requires_grad=False)
    # weighted_mask = weighted_mask[:,None,:,:]

    #===============================================
    # device to CUDA
    #===============================================
    if cuda:
        PET_input = PET_input.cuda()
        anatomical_input = anatomical_input.cuda()
        MRI_full = MRI_full.cuda()
    
    # print(anatomical_input.dtype)
    # print(PET_input.dtype)
    # print(MRI_full.dtype)

    return anatomical_input, PET_input, MRI_full, batch_filname

def initialization():
    #=================================================
    # Set args
    #=================================================
    with open(args.config, 'r') as file:
        configuration = json.load(file)
    for key, value in configuration.items():
        setattr(args, key, value)
    print("Loaded configuration from path: ", args.config)  
    
    #=================================================
    # Check CUDA
    #=================================================
    if args.cuda:
        print("Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    args.seed = random.randint(1, 10000)
    # random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True 

def prepare_data():
    
    inference_dataset=InferenceDataset(args.input_directory, args.suvr, args.segmentation, args.simulation, transform=trns.Compose([trns.ToTensor()]))

    testloader = torch.utils.data.DataLoader(
                    inference_dataset,
                    batch_size=1, shuffle=False)
    return testloader
    
def build_model():

    #===============================================
    # Load backbone model
    #===============================================
    if args.backbone==0:
        input_nc = 1
        output_nc = 1
        nb_filter = [64, 96, 128, 256]
        deepsupervision = False
        
        model_fusion = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=args.fusion, scale_head=4, fusion_op=True) 
    elif args.backbone==1:
        img_size=112
        patch_size=2#1
        in_chans=1
        out_chans=in_chans
        embed_dim=96
        depths=[2, 2, 2, 2]
        num_heads=[3, 6, 12, 24]
        window_size=7
        mlp_ratio=4.
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        drop_path_rate=0.1
        ape=True
        patch_norm=True
        use_checkpoint=False    
        model_fusion=SwinUNet(img_size=img_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            num_classes=out_chans,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate,
                            ape=ape,
                            patch_norm=patch_norm,
                            use_checkpoint=use_checkpoint,
                            fusion_type=args.fusion,
                            fusion_op=True)
    
    if args.fusion_op == False:
        model_conversion = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, scale_head=4, fusion_op=False) 
    else:
        model_conversion=None

    try:
        if args.resume_fusion_checkpoint:
            ckeckpoint_path=str(args.resume_fusion_checkpoint)
            if os.path.isfile(ckeckpoint_path):
                print(torch.load(ckeckpoint_path).keys())
                
                model_fusion.load_state_dict(torch.load(ckeckpoint_path))
            else:
                print("=> no fusion checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        print("AttributeError: 'Namespace' object has no attribute 'resume_fusion_checkpoint'")

    try:
        if args.resume_conversion_checkpoint:
            ckeckpoint_path=str(args.resume_conversion_checkpoint)
            if os.path.isfile(ckeckpoint_path):
                print(torch.load(ckeckpoint_path).keys())
                model_conversion.load_state_dict(torch.load(ckeckpoint_path))
                
            else:
                print("=> no conversion checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        print("AttributeError: 'Namespace' object has no attribute 'resume_conversion_checkpoint'")

    return model_fusion, model_conversion

if __name__ == "__main__":
    global args
    #===============================================
    # Create argument parser
    #===============================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../src/experiments/best.json", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--segmentation", type=int ,default=0, help="No segmentation(0; Default); Pseudo segmentation(1; Default); FreeSurfer segmentation (2); SPM segmentation(3)")
    parser.add_argument("--resume_fusion_checkpoint", default=" ", type=str, help="Path to the pretrained main fusion network")
    parser.add_argument("--resume_conversion_checkpoint", default=" ", type=str, help="Path to the pretrained auxilirary conversion networks")
    parser.add_argument("--input_directory", default="../src/native_space_pet_crop", type=str, help="Path to the input data")
    parser.add_argument("--suvr", action="store_true",default=False, help="Use SUVr (True) or SUV (False; Default)")
    parser.add_argument("--simulation", type=int ,default=0, help="Input type: Real PET(default, 0); Simulation PET using FreeSurfer generated GT (1); Simulation PET using SPM generated GT (2)")


    args = parser.parse_args()
    
    #===============================================
    # Initialization, CUDA Setup, Output Folder Creation
    #===============================================
    initialization()

    testloader = prepare_data()

    model_fusion, model_conversion = build_model()

    model_fusion.eval()
    if args.fusion_op == False:
        model_conversion.eval()

    for iteration, data in enumerate(testloader, 1):
        
    #     #===============================================
    #     # Data preprocessing
    #     #===============================================
        anatomical_input, PET_input, MRI_full, batch_filname = _data_preprocessing(data, args.segmentation, args.cuda)
        
        # print(batch_filname)
        # print("anatmoical input", torch.max(anatomical_input), torch.min(anatomical_input))
        # print("pet input", torch.max(PET_input), torch.min(PET_input))
        
        #===============================================
        # Move model to CUDA
        #===============================================
        if args.cuda:
            model_fusion.cuda()
            if args.resume_conversion_checkpoint != " ":
                model_conversion.cuda()

        #===============================================
        # Foward pass
        #===============================================
        if PET_input.shape[2]==224:
            # print("batch")
            fusion_outputs=[]
            for batch in range(PET_input.shape[0]//16):
                # print(batch)
                fusion_output=model_fusion(PET_input[batch*16:(batch+1)*16], anatomical_input[batch*16:(batch+1)*16]).cpu().detach().numpy()
                # print(fusion_output.shape)
                fusion_outputs.append(fusion_output)
            fusion_outputs=np.concatenate(fusion_outputs, axis=0) 

        else:
            fusion_outputs = model_fusion(PET_input, anatomical_input)
            fusion_outputs = np.asarray(fusion_outputs.cpu().detach().numpy())
        anatomical_input = np.asarray(anatomical_input.cpu().detach().numpy())
        PET_input = np.asarray(PET_input.cpu().detach().numpy())


        if args.resume_conversion_checkpoint != " ":
            PET_preds, MRI_preds = model_conversion(fusion_outputs)
            PET_preds= np.asarray(PET_preds.cpu().detach().numpy())
            MRI_preds= np.asarray(MRI_preds.cpu().detach().numpy())
        # print(fusion_outputs.shape)
        # print(np.max(PET_input), np.min(PET_input))
        fusion_outputs=_data_postprocessing(data, fusion_outputs, anatomical_input)