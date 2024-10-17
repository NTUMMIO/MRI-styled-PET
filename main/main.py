# from utils.mr_styled_pet import MR_styled_PET
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

from utils.data_set import ADNIDataset
from utils.model.backbone.backbone import UNetPP, SwinUNet
from utils.utils import adjust_learning_rate, save_images_per_epoch, save_images_per_epoch_conversion, record_parser, record_search_config, SaveBestModel
from utils.loss import BoundaryGradient, AdaptiveMSSSIM, GradientRegularizer

import matplotlib.pyplot as plt

def tester():

    output_folder = _mk_output_folder(mode=args.test)

    testloader = _prepare_data(mode=args.test)

    model_fusion, model_conversion = _build_model()

    model_fusion.eval()
    if args.fusion_op == False:
        model_conversion.eval()

    for iteration, data in enumerate(testloader, 1):
        #===============================================
        # Data preprocessing
        #===============================================
        anatomical_input, PET_input, MRI_full, weighted_mask, batch_filname = _data_preprocessing(data, args.anatomical_input_option, args.cuda)
        
        # print(PET_input)

        # print(batch_filname)
        # print(anatomical_input.shape)
        # print(PET_input.shape)
        
        #===============================================
        # Move model to CUDA
        #===============================================
        if args.cuda:
            model_fusion.cuda()
            # if args.fusion_op == False:
            #     model_conversion.cuda()

        #===============================================
        # Foward pass
        #===============================================
        fusion_outputs = model_fusion(PET_input, anatomical_input)
        fusion_outputs= np.asarray(fusion_outputs.cpu().detach().numpy())
        anatomical_input= np.asarray(anatomical_input.cpu().detach().numpy())

        # if args.fusion_op == False:
        #     PET_preds, MRI_preds = model_conversion(fusion_outputs)
        #     PET_preds= np.asarray(PET_preds.cpu().detach().numpy())
        #     MRI_preds= np.asarray(MRI_preds.cpu().detach().numpy())
        
        PET_inputs= np.asarray(PET_input.cpu().detach().numpy())
        MRI_fulls= np.asarray(MRI_full.cpu().detach().numpy())
        # anatomical_input= np.asarray(anatomical_input.cpu().detach().numpy())
        
        # print(batch_filname)
        # plt.figure(figsize=(15, 6))
        # plt.subplot(1, 3 , 1)
        # plt.imshow(fusion_outputs[50, 0, :, :])
        # plt.colorbar()
        # plt.axis("off")
        # plt.subplot(1, 3, 2)
        # plt.imshow(fusion_outputs[40, 0, :, :])
        # plt.colorbar()
        # plt.axis("off")
        # plt.subplot(1, 3, 3)
        # plt.imshow(fusion_outputs[30, 0, :, :])
        # plt.colorbar()
        # plt.axis("off")
        # plt.show()
        # plt.savefig(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/output.png"))
            

        for n in range(len(fusion_outputs)):
            input_filepath=batch_filname[n]
            print(input_filepath)
            group_folder=input_filepath.split('/')[5][:-8]
            subject_folder=input_filepath.split('/')[6]
            filename=input_filepath.split('/')[7].replace('.npy', '.tif')
            print(os.path.join(output_folder, "fusion_output", group_folder, subject_folder, filename))
            fusion_output=Image.fromarray(fusion_outputs[n,0,:,:])
            
            anatomical_input_single=Image.fromarray(anatomical_input[n,0,:,:])
            
    

            if not os.path.exists(os.path.join(output_folder, "fusion_output", group_folder, subject_folder)):
                os.makedirs(os.path.join(output_folder, "fusion_output", group_folder, subject_folder))
            fusion_output.save(os.path.join(output_folder, "fusion_output", group_folder, subject_folder, filename))
            
            if not os.path.exists(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/anat/", group_folder, subject_folder)):
                os.makedirs(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/anat/", group_folder, subject_folder))
            anatomical_input_single.save(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/anat/", group_folder, subject_folder, filename))

            if not os.path.exists(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/PET/", group_folder, subject_folder)):
                os.makedirs(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/PET/", group_folder, subject_folder))
                pet_input=Image.fromarray(PET_inputs[n,0,:,:])
                pet_input.save(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/PET/", group_folder, subject_folder, filename))
            
            if not os.path.exists(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/MRI/", group_folder, subject_folder)):
                os.makedirs(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/MRI/", group_folder, subject_folder))
                mri_full=Image.fromarray(MRI_fulls[n,0,:,:])
                mri_full.save(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images/MRI/", group_folder, subject_folder, filename))



            # if args.fusion_op == False:

            #     MRI_pred=Image.fromarray(MRI_preds[n,0,:,:])
            #     PET_pred=Image.fromarray(PET_preds[n,0,:,:])

            #     if not os.path.exists(os.path.join(output_folder, "mri_pred", group_folder, subject_folder)):
            #         os.makedirs(os.path.join(output_folder, "mri_pred", group_folder, subject_folder))
            #     MRI_pred.save(os.path.join(output_folder, "mri_pred", group_folder, subject_folder, filename))

            #     if not os.path.exists(os.path.join(output_folder, "pet_pred", group_folder, subject_folder)):
            #         os.makedirs(os.path.join(output_folder, "pet_pred", group_folder, subject_folder))
            #     PET_pred.save(os.path.join(output_folder, "pet_pred", group_folder, subject_folder, filename))

def trainer(config):

    output_folder = _mk_output_folder(mode=args.test)
    
    trainloader, validloader = _prepare_data(mode=args.test)
    
    model_fusion, model_conversion = _build_model()

    save_best_loss_model = SaveBestModel(name='loss', minmax=0, output_folder=output_folder)

    wandb.init(
    # set the wandb project where this run will be logged
    project=output_folder.split('/')[-4]+"_"+output_folder.split('/')[-3]+"_"+output_folder.split('/')[-2], 
    # track hyperparameters and run metadata
    config=args
    )

    for epoch in range(args.start_epoch, args.nEpochs):

            
    #===============================================    
    # -------> Step 1: Training <-------
    #===============================================
        
        #===============================================
        # Set model to train mode
        #===============================================
        model_fusion.train()
        if args.fusion_op == False:
            model_conversion.train()

        #===============================================
        # Set loss function
        #===============================================
        mse_loss = torch.nn.MSELoss()
        boundary_loss=BoundaryGradient()

        mae_loss=GradientRegularizer(kernel_sizes=[3])
        # mse_loss=GradientRegularizer(mode='SE')
    
        ssim_loss_mri=AdaptiveMSSSIM(data_range=1.0)     
        ssim_loss_pet=AdaptiveMSSSIM(data_range=1.0)  

        for iteration, data in enumerate(trainloader, 1):
            
            lr = adjust_learning_rate(iteration+epoch*len(trainloader), args.lr, args.step)
            #===============================================
            # Set optimizer
            #===============================================
            if args.fusion_op == False:
                optimizer = AdamW([{'params': model_fusion.parameters()}, {'params': model_conversion.parameters()}], lr=lr)   
            else:
                optimizer = AdamW(model_fusion.parameters(), lr)
            
            #===============================================
            # Data preprocessing
            #===============================================
            anatomical_input, PET_input, MRI_full, weighted_mask, batch_filname = _data_preprocessing(data, args.anatomical_input_option, args.cuda)

            #===============================================
            # Move model to CUDA
            #===============================================
            if args.cuda:
                model_fusion.cuda()
                if args.fusion_op == False:
                    model_conversion.cuda()

            #===============================================
            # Foward pass
            #===============================================
            torch.autograd.set_detect_anomaly(True)  
            fusion_output = model_fusion(PET_input, anatomical_input)
            if args.fusion_op == False:
                PET_pred, MRI_pred = model_conversion(fusion_output)

        #===============================================
        # Calculate individual losses
        #===============================================
            mse_loss_PET = mse_loss(fusion_output, PET_input)
            mse_loss_MRI = mse_loss(fusion_output, anatomical_input)
            
            #===============================================
            # Config B: Whether switch SSIM to AdaptiveMSSSIM
            #===============================================
            if args.ssim:
                PET_sets_winSize_weighting=[(7, config['ssim_larger_win_pet']), (5, 1-config['ssim_larger_win_pet']-config['ssim_smaller_win_pet']), (3, config['ssim_smaller_win_pet'])]
                MRI_sets_winSize_weighting=[(7, config['ssim_larger_win_mri']), (5, 1-config['ssim_larger_win_mri']-config['ssim_smaller_win_mri']), (3, config['ssim_smaller_win_mri'])]
                
                ssim_loss_PET = ssim_loss_pet(fusion_output, PET_input, PET_sets_winSize_weighting)
                ssim_loss_MRI = ssim_loss_mri(fusion_output, anatomical_input, MRI_sets_winSize_weighting)
            else:
                ssim_loss_PET=ssim_loss_pet(fusion_output, PET_input)
                ssim_loss_MRI=ssim_loss_mri(fusion_output, anatomical_input)
            
            #===============================================
            # Config C: Whether to include boundary loss
            #===============================================
            if args.boundary:
                # boundary_loss_MRI, boundary_loss_PET =boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)
                boundary_loss_PET=mae_loss(fusion_output, MRI_full, PET_input)
                boundary_loss_MRI=mae_loss(fusion_output, MRI_full, anatomical_input)

            else:
                boundary_loss_MRI=0
                boundary_loss_PET=0

            #===============================================
            # Config E: Whether losses from conversion networks are claculated
            #===============================================
            if args.fusion_op == False:
                mse_loss_PET = mse_loss_PET + mse_loss(PET_pred, PET_input)
                mse_loss_MRI = mse_loss_MRI + mse_loss(MRI_pred, MRI_full)
                
                # boundary_loss_MRI_conversion, boundary_loss_PET_conversion = boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)

                boundary_loss_MRI_conversion=mae_loss(PET_pred, MRI_full, PET_input)
                boundary_loss_PET_conversion=mae_loss(MRI_pred, MRI_full, MRI_full)

                boundary_loss_MRI = boundary_loss_MRI + boundary_loss_MRI_conversion
                boundary_loss_PET = boundary_loss_PET + boundary_loss_PET_conversion

                #===============================================
                # Config B: Whether switch SSIM to AdaptiveMSSSIM
                #===============================================
                if args.ssim:
                    ssim_loss_PET = ssim_loss_PET + ssim_loss_pet(PET_pred, PET_input, PET_sets_winSize_weighting)
                    ssim_loss_MRI = ssim_loss_MRI + ssim_loss_mri(MRI_pred, MRI_full, MRI_sets_winSize_weighting)
                else:
                    ssim_loss_PET=ssim_loss_pet(PET_pred, PET_input)
                    ssim_loss_MRI=ssim_loss_mri(MRI_pred, MRI_full)
                
                #===============================================
                # Config C: Whether to include boundary loss
                #===============================================
                if args.boundary:
                    # boundary_loss_MRI, boundary_loss_PET =boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)
                    boundary_loss_PET=mae_loss(PET_pred, MRI_full, PET_input)
                    boundary_loss_MRI=mae_loss(MRI_pred, MRI_full, MRI_full)
                else:
                    boundary_loss_MRI=0
                    boundary_loss_PET=0

            #===============================================
            # Calculate total loss
            #===============================================
            total_loss = \
                mse_loss_PET*args.pixel_pet2mri + \
                mse_loss_MRI + \
                boundary_loss_PET*args.boundary_pet2mri + \
                boundary_loss_MRI + \
                ssim_loss_PET*args.ssim_pet2mri + \
                ssim_loss_MRI 
            
            #===============================================    
            # Compute gradient
            #===============================================    
            optimizer.zero_grad()

            #===============================================    
            # Backward pass
            #===============================================  
            # with torch.autograd.detect_anomaly():
            #     total_loss.backward(retain_graph=True)
            total_loss.backward()

            #===============================================    
            # Report loss to RayTune
            #===============================================
            #===============================================    
            # Report loss to WandB
            #===============================================
            if args.boundary:
                ray.train.report({'total_loss': total_loss.item(), 
                    'MSE loss (PET)': mse_loss_PET.item(), 
                    'MSE loss (MRI)': mse_loss_MRI.item(), 
                    'SSIM loss (PET)': ssim_loss_PET.item(), 
                    'SSIM loss (MRI)': ssim_loss_MRI.item(), 
                    'Tissue-aware loss (PET)': boundary_loss_MRI.item(), 
                    'Tissue-aware loss (MRI)': boundary_loss_PET.item()})
                
                wandb.log({'Train: Total loss': total_loss.item(), 
                    'Train: MSE loss (PET)': mse_loss_PET.item(), 
                    'Train: MSE loss (MRI)': mse_loss_MRI.item(), 
                    'Train: SSIM loss (PET)': ssim_loss_PET.item(), 
                    'Train: SSIM loss (MRI)': ssim_loss_MRI.item(), 
                    'Train: Tissue-aware loss (PET)': boundary_loss_MRI.item(), 
                    'Train: Tissue-aware loss (MRI)': boundary_loss_PET.item(),
                    'Learing rate': lr})
                
            else:
                ray.train.report({'total_loss': total_loss.item(), 
                    'MSE loss (PET)': mse_loss_PET.item(), 
                    'MSE loss (MRI)': mse_loss_MRI.item(), 
                    'SSIM loss (PET)': ssim_loss_PET.item(), 
                    'SSIM loss (MRI)': ssim_loss_MRI.item()})
                
                wandb.log({'Train: Total loss': total_loss.item(), 
                    'Train: MSE loss (PET)': mse_loss_PET.item(), 
                    'Train: MSE loss (MRI)': mse_loss_MRI.item(), 
                    'Train: SSIM loss (PET)': ssim_loss_PET.item(), 
                    'Train: SSIM loss (MRI)': ssim_loss_MRI.item(),
                    'Learing rate': lr})
            
            #===============================================    
            # Update weights
            #===============================================
            optimizer.step()

            #===============================================
            # Save images every args.save_iter
            #===============================================
            if iteration % args.save_iter == 0:
                if args.fusion_op == False:
                    save_images_per_epoch_conversion(MRI_full.cpu().detach().numpy(), anatomical_input.cpu().detach().numpy(), PET_input.cpu().detach().numpy(), fusion_output.cpu().detach().numpy(), MRI_pred.cpu().detach().numpy(), PET_pred.cpu().detach().numpy(), output_folder, epoch, True, iteration, data[2])
                else:
                    save_images_per_epoch(MRI_full.cpu().detach().numpy(), anatomical_input.cpu().detach().numpy(), PET_input.cpu().detach().numpy(), fusion_output.cpu().detach().numpy(), output_folder, epoch, True, iteration, data[2])
            # break
    #===============================================    
    # -------> Step 2: Validation <-------
    #===============================================
        
        model_fusion.eval()
        if args.fusion_op == False:
            model_conversion.eval()

        #===============================================
        # Set loss function
        #===============================================
        mse_loss = torch.nn.MSELoss()
        boundary_loss=BoundaryGradient()

        mae_loss=GradientRegularizer(kernel_sizes=[3])
        # mse_loss=GradientRegularizer(mode='SE')

        ssim_loss_mri=AdaptiveMSSSIM(data_range=1.0)     
        ssim_loss_pet=AdaptiveMSSSIM(data_range=1.0) 
        

        for iteration, data in enumerate(validloader, 1):
            #===============================================
            # Data preprocessing
            #===============================================
            anatomical_input, PET_input, MRI_full, weighted_mask, batch_filname = _data_preprocessing(data, args.anatomical_input_option, args.cuda)

            #===============================================
            # Move model to CUDA
            #===============================================
            if args.cuda:
                model_fusion.cuda()
                if args.fusion_op == False:
                    model_conversion.cuda()

            #===============================================
            # Foward pass
            #===============================================
            torch.autograd.set_detect_anomaly(True)  
            fusion_output = model_fusion(PET_input, anatomical_input)
            if args.fusion_op == False:
                PET_pred, MRI_pred = model_conversion(fusion_output)

        #===============================================
        # Calculate individual losses
        #===============================================
            mse_loss_PET = mse_loss(fusion_output, PET_input)
            mse_loss_MRI = mse_loss(fusion_output, anatomical_input)
            
            #===============================================
            # Config B: Whether switch SSIM to AdaptiveMSSSIM
            #===============================================
            if args.ssim:
                PET_sets_winSize_weighting=[(7, config['ssim_larger_win_pet']), (5, 1-config['ssim_larger_win_pet']-config['ssim_smaller_win_pet']), (3, config['ssim_smaller_win_pet'])]
                MRI_sets_winSize_weighting=[(7, config['ssim_larger_win_mri']), (5, 1-config['ssim_larger_win_mri']-config['ssim_smaller_win_mri']), (3, config['ssim_smaller_win_mri'])]
                
                ssim_loss_PET = ssim_loss_pet(fusion_output, PET_input, PET_sets_winSize_weighting)
                ssim_loss_MRI = ssim_loss_mri(fusion_output, anatomical_input, MRI_sets_winSize_weighting)
            else:
                ssim_loss_PET=ssim_loss_pet(fusion_output, PET_input)
                ssim_loss_MRI=ssim_loss_mri(fusion_output, anatomical_input)
            
            #===============================================
            # Config C: Whether to include boundary loss
            #===============================================
            if args.boundary:
                # boundary_loss_MRI, boundary_loss_PET =boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)
                boundary_loss_PET=mae_loss(fusion_output, MRI_full, PET_input)
                boundary_loss_MRI=mae_loss(fusion_output, MRI_full, anatomical_input)

            else:
                boundary_loss_MRI=0
                boundary_loss_PET=0

            #===============================================
            # Config E: Whether losses from conversion networks are claculated
            #===============================================
            if args.fusion_op == False:
                mse_loss_PET = mse_loss_PET + mse_loss(PET_pred, PET_input)
                mse_loss_MRI = mse_loss_MRI + mse_loss(MRI_pred, MRI_full)
                
                # boundary_loss_MRI_conversion, boundary_loss_PET_conversion = boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)

                boundary_loss_MRI_conversion=mae_loss(PET_pred, MRI_full, PET_input)
                boundary_loss_PET_conversion=mae_loss(MRI_pred, MRI_full, MRI_full)

                boundary_loss_MRI = boundary_loss_MRI + boundary_loss_MRI_conversion
                boundary_loss_PET = boundary_loss_PET + boundary_loss_PET_conversion

                #===============================================
                # Config B: Whether switch SSIM to AdaptiveMSSSIM
                #===============================================
                if args.ssim:
                    ssim_loss_PET = ssim_loss_PET + ssim_loss_pet(PET_pred, PET_input, PET_sets_winSize_weighting)
                    ssim_loss_MRI = ssim_loss_MRI + ssim_loss_mri(MRI_pred, MRI_full, MRI_sets_winSize_weighting)
                else:
                    ssim_loss_PET=ssim_loss_pet(PET_pred, PET_input)
                    ssim_loss_MRI=ssim_loss_mri(MRI_pred, MRI_full)
                
                #===============================================
                # Config C: Whether to include boundary loss
                #===============================================
                if args.boundary:
                    # boundary_loss_MRI, boundary_loss_PET =boundary_loss(fusion_output, PET_input, anatomical_input, weighted_mask)
                    boundary_loss_PET=mae_loss(PET_pred, MRI_full, PET_input)
                    boundary_loss_MRI=mae_loss(MRI_pred, MRI_full, MRI_full)
                else:
                    boundary_loss_MRI=0
                    boundary_loss_PET=0

            #===============================================
            # Calculate total loss
            #===============================================
            total_loss = \
                mse_loss_PET*args.pixel_pet2mri + \
                mse_loss_MRI + \
                boundary_loss_PET*args.boundary_pet2mri + \
                boundary_loss_MRI + \
                ssim_loss_PET*args.ssim_pet2mri + \
                ssim_loss_MRI 

            #===============================================    
            # Report loss to WandB
            #===============================================
            if args.boundary:
                wandb.log({'Valid: Total loss': total_loss.item(), 
                    'Valid: MSE loss (PET)': mse_loss_PET.item(), 
                    'Valid: MSE loss (MRI)': mse_loss_MRI.item(), 
                    'Valid: SSIM loss (PET)': ssim_loss_PET.item(), 
                    'Valid: SSIM loss (MRI)': ssim_loss_MRI.item(), 
                    'Valid: Tissue-aware loss (PET)': boundary_loss_MRI.item(), 
                    'Valid: Tissue-aware loss (MRI)': boundary_loss_PET.item()})
            else:
                wandb.log({'Valid: Total loss': total_loss.item(), 
                    'Valid: MSE loss (PET)': mse_loss_PET.item(), 
                    'Valid: MSE loss (MRI)': mse_loss_MRI.item(), 
                    'Valid: SSIM loss (PET)': ssim_loss_PET.item(), 
                    'Valid: SSIM loss (MRI)': ssim_loss_MRI.item()})
            
            # break
        save_best_loss_model(total_loss.item(),epoch, args.fold, model_fusion, model_conversion)
        record_search_config(config, output_folder)
    
    wandb.finish()

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

def _mk_output_folder(mode):

    if mode==True:
    #===============================================
    # Testing mode
    #===============================================
        #=================================================
        # Create corresponding output folder with checkpoint path
        #=================================================
        output_folder=args.resume_fusion_checkpoint.replace('model_checkpoint', 'output_images').replace('checkpoint/model_fusion_best_loss_fold_0.pth', '')
    else:
    #===============================================
    # Training mode
    #===============================================
        #=================================================
        # Assign output folder with current timestamps
        #=================================================
        output_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/model_checkpoint/"+args.folder+"/fold"+str(args.fold)+"/"+str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('Output folder: ', output_folder)

    #=================================================
    # Save parset arguments
    #=================================================
    record_parser(output_folder, args)

    return output_folder

def _prepare_data(mode):
    if mode==True:
    #===============================================
    # Testing mode
    #===============================================
        MRI_test_filenames=[]    
        PET_test_filenames=[]
        
        #===============================================
        # Load MRI testing data  
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/test_MRI.txt'.format(str(args.fold)), 'r')  
        for line in f.readlines():
            MRI_test_filenames.append('/home/linyunong/project/'+str(line)[:-2])
            
        #===============================================
        # Load PET testing data   
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/test_PET.txt'.format(str(args.fold)), 'r')
        for line in f.readlines():
            PET_test_filenames.append('/home/linyunong/project/'+str(line)[:-2])

        #===============================================
        # ADNIDataset and DataLoader
        #===============================================
        test_dataset = ADNIDataset(root0=PET_test_filenames, root1=MRI_test_filenames, inputScale=args.inputScale, transform=trns.Compose([trns.ToTensor()]))
        

        testloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.batchSize, shuffle=False)
        return testloader

    else:
    #===============================================
    # Training mode
    #===============================================

        MRI_train_filenames=[]    
        PET_train_filenames=[]
        MRI_valid_filenames=[]    
        PET_valid_filenames=[]
        
        #===============================================
        # Load MRI training data  
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/train_MRI.txt'.format(str(args.fold)), 'r')  
        for line in f.readlines():
            MRI_train_filenames.append('/home/linyunong/project/'+str(line)[:-2])
            
        #===============================================
        # Load PET training data   
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/train_PET.txt'.format(str(args.fold)), 'r')
        for line in f.readlines():
            PET_train_filenames.append('/home/linyunong/project/'+str(line)[:-2])

        #===============================================
        # Load MRI validation data  
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/valid_MRI.txt'.format(str(args.fold)), 'r')  
        for line in f.readlines():
            MRI_valid_filenames.append('/home/linyunong/project/'+str(line)[:-2])
            
        #===============================================
        # Load PET validation data   
        #===============================================
        f = open('/home/linyunong/project/src/fold_750/fold{}/valid_PET.txt'.format(str(args.fold)), 'r')
        for line in f.readlines():
            PET_valid_filenames.append('/home/linyunong/project/'+str(line)[:-2])
            
        #===============================================
        # ADNIDataset and DataLoader
        #===============================================
        train_dataset = ADNIDataset(root0=PET_train_filenames, root1=MRI_train_filenames, inputScale=args.inputScale, transform=trns.Compose([trns.ToTensor()]))
        valid_dataset = ADNIDataset(root0=PET_valid_filenames, root1=MRI_valid_filenames, inputScale=args.inputScale, transform=trns.Compose([trns.ToTensor()]))
        
        trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=args.batchSize, shuffle=True)
        
        validloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=args.batchSize, shuffle=True)
        return trainloader, validloader
    
def _build_model():

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
                print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        
        print("AttributeError: 'Namespace' object has no attribute 'resume_fusion_checkpoint'")

    try:
        if args.resume_conversion_checkpoint:
            ckeckpoint_path=str(args.resume_conversion_checkpoint)
            if os.path.isfile(ckeckpoint_path):
                print(torch.load(ckeckpoint_path).keys())
                model_conversion.load_state_dict(torch.load(ckeckpoint_path))
                
            else:
                print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        print("AttributeError: 'Namespace' object has no attribute 'resume_conversion_checkpoint'")

    return model_fusion, model_conversion

# def _z_score_normalize(image):
#     mean = torch.mean(image[image != 0])
#     std = torch.std(image[image != 0])
    
#     # Normalize the image
#     normalized_image = (image - mean) / (std + 1e-5)  # Add a small value to avoid division by zero
    
#     return normalized_image

# def _reverse_min_max_normalize(image):

#     # Compute min and max values (excluding zero pixels)
#     min_val = torch.min(image)
#     max_val = torch.max(image)
    
#     # Min-max normalization formula
#     normalized_image = (image - min_val) / (max_val - min_val + 1e-5)  # Add small epsilon to avoid division by zero
    
#     return 1-normalized_image

def _z_score_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """

    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    std = torch.std(image, dim=(1, 2), keepdim=True)
    return (image - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero

def _reverse_min_max_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """
    
    # non_zero_mask = (image != 0)  # Create a mask where non-zero pixels are True
    # masked_image = torch.where(non_zero_mask, image, torch.tensor(float('inf')))

    min_val = image.amin(dim=(1, 2), keepdim=True)#torch.min(image, dim=(1, 2), keepdim=True)
    # print(min_val.shape)
    max_val = image.amax(dim=(1, 2), keepdim=True)#torch.max(image, dim=(1, 2), keepdim=True)
    # print(max_val.shape)
    
    return torch.ones(image.shape) - (image - min_val) / (max_val - min_val + 1e-8)


def _min_max_normalize(image):
    """
    Normalizes a single image (C, H, W) on a channel-wise basis.
    """
    
    # non_zero_mask = (image != 0)  # Create a mask where non-zero pixels are True
    # masked_image = torch.where(non_zero_mask, image, torch.tensor(float('inf')))
    
    min_val = image.amin(dim=(1, 2), keepdim=True)#torch.min(image, dim=(1, 2), keepdim=True)
    # print(min_val.shape)
    max_val = image.amax(dim=(1, 2), keepdim=True)#torch.max(image, dim=(1, 2), keepdim=True)
    # print(max_val.shape)
    
    return (image - min_val) / (max_val - min_val + 1e-8)

def _logistic_transform(x, a=5):
    return torch.pow(x, a) / (torch.pow(x, a) + torch.pow(1 - x, a))

def _data_preprocessing(data, anatomical_input_option=3, cuda=True):

    batch_pet=data[0]
    batch_mri=data[1]
    batch_filname=data[2]

    #===============================================
    # PET input
    #===============================================
    PET_input = Variable(batch_pet[:,0,:,:],requires_grad=True)
    PET_input = PET_input[:,None,:,:]
    # print(PET_input.shape, torch.max(PET_input), torch.min(PET_input))
    #===============================================
    # Full T1-MRI
    #===============================================
    MRI_full = Variable(batch_mri[:,0,:,:], requires_grad=True)
    MRI_full = MRI_full[:,None,:,:]
    
    #===============================================
    # Anatomical input
    # 0 = full MRI
    # 1 = PET coarse
    # 2 = PET fine   
    # 3 = = predefined 4:1:0
    # 4 = predefined 5:1:0
    # 5 = GTM coarse
    # 6 = GTM fine
    # 7 = MR-derived intensity feature
    #===============================================
    # if anatomical_input_option==7:
        
    #     anatomical_input=Variable(batch_mri[:,0,:,:], requires_grad=True)
    #     print(anatomical_input.shape)
        # print(anatomical_input.shape)
        
        # csf_mask = torch.where(anatomical_input < 0.2, torch.tensor(0.0), torch.tensor(1.0))
        
        # anatomical_input=_reverse_min_max_normalize(torch.abs(_z_score_normalize(anatomical_input)))
        # anatomical_input=anatomical_input*csf_mask
      
        # anatomical_input=_min_max_normalize(_z_score_normalize(anatomical_input))
        
        # previous version
        # mask=torch.where(anatomical_input<0.2, torch.tensor(0.0), torch.tensor(1.0))
        # anatomical_input=anatomical_input*mask
        # anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        # anatomical_input=anatomical_input*mask
        
        # mask=torch.where(anatomical_input<0.3, torch.tensor(0.0), torch.tensor(1.0))
        # anatomical_input=anatomical_input*mask
        # anatomical_input=_reverse_min_max_normalize(anatomical_input)  
        # anatomical_input=anatomical_input*mask
        # anatomical_input=_min_max_normalize(anatomical_input)  
        # anatomical_input=_logistic_transform(anatomical_input, a=1.5) 
    
    # print(batch_filname)
    anatomical_input=Variable(batch_mri[:,anatomical_input_option,:,:], requires_grad=True)
    # print(anatomical_input.shape)    
    # print(torch.max(batch_mri[10,anatomical_input_option,:,:]), torch.min(batch_mri[10,anatomical_input_option,:,:]))
    
    
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 3 , 1)
    # plt.imshow(anatomical_input.detach().numpy()[50, :, :])
    # plt.colorbar()
    # plt.axis("off")
    # plt.subplot(1, 3, 2)
    # plt.imshow(anatomical_input.detach().numpy()[40, :, :])
    # plt.colorbar()
    # plt.axis("off")
    # plt.subplot(1, 3, 3)
    # plt.imshow(anatomical_input.detach().numpy()[30, :, :])
    # plt.colorbar()
    # plt.axis("off")
    # plt.show()
    # plt.savefig(os.path.join("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/main/experiment_seg.png"))
    
          
        
        # anatomical_input = _z_score_normalize(anatomical_input)
        # anatomical_input = torch.where(anatomical_input < 0, torch.tensor(0.0), anatomical_input)
        # anatomical_input=1/(anatomical_input+ 1e-8)
        # anatomical_input = _min_max_normalize(-anatomical_input)
        # anatomical_input = torch.ones(anatomical_input.shape)/torch.sqrt(anatomical_input)
        # anatomical_input = _z_score_normalize(anatomical_input)
        # anatomical_input = torch.abs(anatomical_input)
        # print(anatomical_input.shape)

    anatomical_input = anatomical_input[:,None,:,:]
    # print(anatomical_input.shape)
    #===============================================
    # Weighted mask
    #===============================================
    weighted_mask = Variable(batch_mri[:,3,:,:], requires_grad=True)
    weighted_mask = weighted_mask[:,None,:,:]

    #===============================================
    # device to CUDA
    #===============================================
    if cuda:
        PET_input = PET_input.cuda()
        anatomical_input = anatomical_input.cuda()
        MRI_full = MRI_full.cuda()
        weighted_mask = weighted_mask.cuda()

    return anatomical_input, PET_input, MRI_full, weighted_mask, batch_filname

if __name__ == "__main__":
    global args
    #===============================================
    # Create argument parser
    #===============================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--fold", default="0", type=str, help="Current fold")
    parser.add_argument("--test", action="store_true",default=False, help="Enter testing mode")
    parser.add_argument("--resume_fusion_checkpoint", default=" ", type=str, help="Path to the pretrained main fusion network")
    parser.add_argument("--resume_conversion_checkpoint", default=" ", type=str, help="Path to the pretrained auxilirary conversion networks")
    parser.add_argument("--inputScale", action="store_true",default=False, help="inputScale=True (image shape 224*224; otherwise, 112*112)")

    args = parser.parse_args()
    
    #===============================================
    # Initialization, CUDA Setup, Output Folder Creation
    #===============================================
    initialization()
    
    #===============================================
    # Determine to enter training/testing mode
    #===============================================
    if args.test:
        tester()

    else:
        #===============================================
        # Init RayTune
        #===============================================
        ray.init(num_cpus=1, num_gpus=1, local_mode=True)

        #===============================================
        # Set up search space for tunable params
        #===============================================
        
        config = {
            "ssim_smaller_win_pet": tune.loguniform(0.3, 0.5), #(0.1, 0.5)
            "ssim_larger_win_pet": tune.loguniform(0.1, 0.3), #(0.1, 0.4)
            
            "ssim_smaller_win_mri": tune.loguniform(0.3, 0.5),
            "ssim_larger_win_mri": tune.loguniform(0.1, 0.3),
        }
        
        #===============================================
        # Start Raytune by passing trainer function and config
        #===============================================
        analysis = tune.run(
            trainer,
            config=config,
            resources_per_trial={"cpu": 1, "gpu": 1},
            metric="total_loss",
            mode="min",  # Use "min" mode for minimizing the loss
        )
        #===============================================
        # Print out the best hyperparameters of this epoch
        #===============================================
        best_config = analysis.get_best_config(metric="total_loss", mode="min")
        print("Best hyperparameters:", best_config)

        # Optionally, you can print more information about the analysis, such as the best trial
        # print("Best trial:")
        # print(analysis.get_best_trial(metric="total_loss", mode="min"))
        
        #===============================================
        # Shut down Ray
        #===============================================
        ray.shutdown()
    
    
    # mr_styled_pet=MR_styled_PET(args=args)
    # print(mr_styled_pet.get_args())
    # print(mr_styled_pet.get_output_folder())
    # mr_styled_pet.prepare_data()
    # mr_styled_pet.build_model()
    # mr_styled_pet.start_training()
    # mr_styled_pet.run_validation()
    