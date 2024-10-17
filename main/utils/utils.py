import numpy as np
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import json
import torch
#===============================================
# Load the pre-split training and validation dataset for each fold
# for old dataset, CN: 100 (rotated train/valid/test) ->fold_112_slice_norm
# for new dataset, test: 50/50/50, train and valid: 5 fold CV for 200/200/200 ->fold_750
#===============================================
# def load_data_path_per_fold(mri_filename, pet_filename, new_dataset, fusion_folder=None, MRI_folder=None):
#     def map_category(input_string):
#         if 'CN' in input_string:
#             return 1.0, 0.0, 0.0
#         elif 'MCI' in input_string:
#             return 0.0, 1.0, 0.0
#         elif 'AD' in input_string:
#             return 0.0, 0.0, 1.0
    
#     MRI=[]
#     PET=[] 
#     MRI_inverted=[]
#     MRI_prior=[]
#     fusion=[]
#     MRI_input=[]
#     group=[]
#     #===============================================
#     # Use old or new dataset
#     #===============================================

# # /home/linyunong/project/style_transfer/fusion_train_images/sc_att_prior/fold0/2024-1-2-94110/AD_FDG_250/002_S_0729.nii/002_S_0729_Axial_7.npy
# # /home/linyunong/project/style_transfer/fusion_train_images/src/fold0/MRI_input/AD_FDG_250/002_S_0729.nii/002_S_0729_Axial_7.npy
# # src/CN_MRI_250/024_S_4084.nii/024_S_4084_Coronal_17.npy'
#     if new_dataset:
#         mri_filename=mri_filename.replace('fold_112_slice_norm/', 'fold_750/')
#         pet_filename=pet_filename.replace('fold_112_slice_norm/', 'fold_750/')
#         mri_inverted_filename = mri_filename.replace('fold_112_slice_norm/', 'fold_750/')
#         mri_prior_filename = mri_filename.replace('fold_112_slice_norm/', 'fold_750/')
#     #===============================================
#     # generate MRI path    
#     #===============================================
#     f = open(mri_filename, 'r')  
#     for line in f.readlines():
#         MRI.append(str(line)[:-2])
        
#     #===============================================
#     # generate PET path    
#     #===============================================
#     f = open(pet_filename, 'r')
#     for line in f.readlines():
#         PET.append(str(line)[:-2])
#         if MRI_folder!=None and fusion_folder!=None:
#             MRI_input.append(str(line)[:-2].replace('src', MRI_folder))
#             fusion.append(str(line)[:-2].replace('src', fusion_folder))
#     #===============================================
#     # generate inverted MRI path    
#     #===============================================
#     f = open(mri_inverted_filename, 'r')  
#     for line in f.readlines():
#         line_in=str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/')
#         line_in=line_in.replace('_250/', '_250_inverted/')
#         MRI_inverted.append(line_in)
#     #===============================================
#     # generate prior MRI path    
#     #===============================================
#     f = open(mri_prior_filename, 'r')  
#     for line in f.readlines():
#         line_in=str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/')
#         line_in=line_in.replace('_250/', '_250_prior_FS_wCerebellum/') #_250_prior2
#         MRI_prior.append(line_in)
#         group.append(map_category(line_in))
#     group=np.array(group)#np.swapaxes(np.array(group), 1, 0)
#     # print(group.shape)    
#     if fusion_folder!=None and MRI_folder!=None:   
#         return MRI, PET, MRI_input, fusion, group
#     else:
#         return MRI, PET, MRI_inverted, MRI_prior, group


#===============================================
# Load the pre-split training and validation dataset for each fold
# for old dataset, CN: 100 (rotated train/valid/test) ->fold_112_slice_norm
# for new dataset, test: 50/50/50, train and valid: 5 fold CV for 200/200/200 ->fold_750
#===============================================

# def load_data_path_per_fold(mri_filename, pet_filename, new_dataset):
    
#     MRI=[]
#     PET=[] 
#     #===============================================
#     # generate MRI path    
#     #===============================================
#     f = open(mri_filename, 'r')  
#     for line in f.readlines():
#         MRI.append(str(line)[:-2].replace('_250', '_input_experiment'))
        
#     #===============================================
#     # generate PET path    
#     #===============================================
#     f = open(pet_filename, 'r')
#     for line in f.readlines():
#         PET.append(str(line)[:-2])
    
#     return MRI, PET
    
    
def adjust_learning_rate(steps, baseline_lr, step):
    """Sets the learning rate to the initial LR decayed by 10"""
    if steps< 7500:
        lr = baseline_lr * (0.95 ** (steps // step))#0.1
    else:
        lr = baseline_lr * (0.98 ** (steps // step))


    return lr 

def psnr(target, ref, scale=None):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    diff = diff.flatten()
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)


#===============================================
# Collect the right channel for input
#===============================================
# def prep_data(data, args):
#     batch=data[0]
#     gt=data[2]
#     # batch_class=np.array([item.split('_')[0].split('/')[-1] for item in data[1]])
#     #print(batch_path)
#     #print(batch.shape)
    
#     #===============================================
#     # PET
#     #===============================================
#     PET_input = Variable(batch[:,0,0,:,:],requires_grad=True)
#     PET_input = PET_input[:,None,:,:]
#     PET_input = PET_input*args.pet_ratio
    
#     #===============================================
#     # MRI
#     #===============================================
#     MRI_full = Variable(batch[:,1,0,:,:], requires_grad=True)
#     MRI_full = MRI_full[:,None,:,:]
#     MRI_full = MRI_full*args.mri_ratio
    
#     #===============================================
#     # MRI input
#     #===============================================
#     # input = full MRI
#     if args.mri_input==0:
#         MRI_input = Variable(batch[:,1,0,:,:], requires_grad=True)
#     # input = gray matter MRI    
#     elif args.mri_input==1:    
#         MRI_input = Variable(batch[:,1,1,:,:], requires_grad=True)
#     # input = full MRI, inverted
#     elif args.mri_input==2:   
#         MRI_input = Variable(batch[:,2,0,:,:], requires_grad=True)
#     # input = gray matter MRI, inverted
#     elif args.mri_input==3:   
#         MRI_input = Variable(batch[:,2,1,:,:], requires_grad=True)
#     elif args.mri_input==4:
#         # print("prior")
#         MRI_input = Variable(batch[:,3,2,:,:], requires_grad=True)
        
#     MRI_input = MRI_input[:,None,:,:]
#     MRI_input = MRI_input*args.mri_ratio
    
#     #===============================================
#     # device to CUDA
#     #===============================================
#     if args.cuda:

#         PET_input = PET_input.cuda()
#         MRI_input = MRI_input.cuda()
#         MRI_full = MRI_full.cuda()

#     return MRI_input, PET_input, MRI_full, gt

def prep_data(data, args):
    batch_pet=data[0]
    batch_mri=data[1]
    # print(data[2])
    # print(batch_mri.shape)
    # print(batch_pet.shape)# gt=data[2]
    # batch_class=np.array([item.split('_')[0].split('/')[-1] for item in data[1]])
    #print(batch_path)
    #print(batch.shape)
    
    #===============================================
    # PET
    #===============================================
    PET_input = Variable(batch_pet[:,0,:,:],requires_grad=True)
    PET_input = PET_input[:,None,:,:]
    PET_input = PET_input*args.pet_ratio
    
    #===============================================
    # MRI
    #===============================================
    MRI_full = Variable(batch_mri[:,0,:,:], requires_grad=True)
    MRI_full = MRI_full[:,None,:,:]
    MRI_full = MRI_full*args.mri_ratio
    
    #===============================================
    # MRI input
    #===============================================
    # 1. input = full MRI
    if args.mri_input==0:    
        MRI_input = Variable(batch_mri[:,0,:,:], requires_grad=True)
    # 2. input = PET coarse
    elif args.mri_input==1:
        MRI_input = Variable(batch_mri[:,1,:,:], requires_grad=True)
    # input = PET fine   
    elif args.mri_input==2:    
        MRI_input = Variable(batch_mri[:,2,:,:], requires_grad=True)
    # input = predefined 4:1:0
    elif args.mri_input==3:   
        MRI_input = Variable(batch_mri[:,3,:,:], requires_grad=True)
    # input = predefined 5:1:0
    elif args.mri_input==4:   
        MRI_input = Variable(batch_mri[:,4,:,:], requires_grad=True)
    # input = GTM coarse
    elif args.mri_input==5:
        MRI_input = Variable(batch_mri[:,5,:,:], requires_grad=True)
    # input = GTM fine
    elif args.mri_input==6:
        MRI_input = Variable(batch_mri[:,6,:,:], requires_grad=True)
        
    MRI_input = MRI_input[:,None,:,:]
    MRI_input = MRI_input*args.mri_ratio
    #===============================================
    # MRI input
    #===============================================
    weighted_mask = Variable(batch_mri[:,3,:,:], requires_grad=True)
    #===============================================
    # device to CUDA
    #===============================================
    if args.cuda:

        PET_input = PET_input.cuda()
        MRI_input = MRI_input.cuda()
        MRI_full = MRI_full.cuda()
        weighted_mask = weighted_mask.cuda()

    return MRI_input, PET_input, MRI_full, weighted_mask#, gt

def prep_conversion_data(data, args):
    batch=data[0]
    gt=data[2]
    
    # batch_path=data[1]
    # print("batch", batch.shape)
    #print(batch_path)
    #print(batch.shape)
    # PET_train, root1=MRI_train, root2=MRI_input_train, root3=fusion_train
    #===============================================
    # PET
    #===============================================
    PET_input = Variable(batch[:,0,0,:,:],requires_grad=True)
    PET_input = PET_input[:,None,:,:]
    PET_input = PET_input*args.pet_ratio
    
    #===============================================
    # MRI
    #===============================================
    MRI_full = Variable(batch[:,1,0,:,:], requires_grad=True)
    MRI_full = MRI_full[:,None,:,:]
    MRI_full = MRI_full*args.mri_ratio
    
    #===============================================
    # MRI input
    MRI_input = Variable(batch[:,2,0,:,:], requires_grad=True)
    MRI_input = MRI_input[:,None,:,:]
    MRI_input = MRI_input*args.mri_ratio
    
    #===============================================
    # fusion
    fusion = Variable(batch[:,3,0,:,:], requires_grad=True)
    fusion = fusion[:,None,:,:]
    
    #===============================================
    # device to CUDA
    #===============================================
    if args.cuda:

        PET_input = PET_input.cuda()
        MRI_full = MRI_full.cuda()
        MRI_input = MRI_input.cuda()
        fusion = fusion.cuda()

    return PET_input, MRI_full, MRI_input, fusion, gt

def save_images_per_epoch(MRI_full, anatomical_input, PET_input, outputs, output_folder, epoch, state, iter, subjects):
    plt.figure(figsize=(30,30))  
    # print(MRI_full.shape)
    for i in range(4):
        mri_min=np.min([np.min(MRI_full[i,0]), np.min(anatomical_input[i,0])])
        mri_max=np.max([np.max(MRI_full[i,0]), np.max(anatomical_input[i,0])])
        pet_min=np.min([np.min(outputs[i,0]), np.min(PET_input[i,0])])
        pet_max=np.max([np.max(outputs[i,0]), np.max(PET_input[i,0])])
        
        # mri_min=np.minimum(MRI_full[i,0], MRI_input[i,0])
        # mri_max=np.maximum(MRI_full[i,0], MRI_input[i,0])
        # pet_min=np.minimum(outputs[i,0], PET_input[i,0])
        # pet_max=np.maximum(outputs[i,0], PET_input[i,0])
        # print(mri_min, mri_max, pet_min, pet_max)
        
        plt.subplot(4,4,4*i+1)
        plt.axis('off')
        plt.imshow(MRI_full[i,0],cmap='gray', vmin=mri_min, vmax=mri_max)
        plt.colorbar()
        plt.title("Full MRI ("+subjects[i].split('/')[-3]+"_"+subjects[i].split('/')[-1]+")")
        
        plt.subplot(4,4,4*i+2)
        plt.axis('off')
        plt.imshow(anatomical_input[i,0],cmap='gray', vmin=mri_min, vmax=mri_max)
        plt.colorbar()
        plt.title("Anatomical input")
        
        plt.subplot(4,4,4*i+3)
        plt.axis('off')
        plt.imshow(PET_input[i,0], cmap='jet', vmin=pet_min, vmax=pet_max)
        plt.colorbar()
        plt.title("PET input")
        
        plt.subplot(4,4,4*i+4)
        plt.axis('off')
        plt.imshow(outputs[i,0], cmap='jet', vmin=pet_min, vmax=pet_max)
        plt.colorbar()
        plt.title("Fusion output")
    if state:    
        filename="train_epoch"+str(epoch)+"_"+str(iter)+".png"
    else:
        filename="valid_epoch"+str(epoch)+"_"+str(iter)+".png"
        
    filefolder=output_folder+"output_images/"

    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()
    
def save_images_per_epoch_conversion(MRI_full, anatomical_input, PET_input, fusion, MRI_pred, PET_pred, output_folder, epoch, state, iter, subjects):
    plt.figure(figsize=(30,30))  
    # print(MRI_full.shape)
    for i in range(4):
        mri_min=np.min([np.min(MRI_full[i,0]), np.min(anatomical_input[i,0]), np.min(MRI_pred[i,0])])
        mri_max=np.max([np.max(MRI_full[i,0]), np.max(anatomical_input[i,0]), np.max(MRI_pred[i,0])])
        pet_min=np.min([np.min(PET_pred[i,0]), np.min(PET_input[i,0]), np.min(fusion[i,0])])
        pet_max=np.max([np.max(PET_pred[i,0]), np.max(PET_input[i,0]), np.max(fusion[i,0])])
        
        # mri_min=np.minimum(MRI_full[i,0], MRI_input[i,0])
        # mri_max=np.maximum(MRI_full[i,0], MRI_input[i,0])
        # pet_min=np.minimum(outputs[i,0], PET_input[i,0])
        # pet_max=np.maximum(outputs[i,0], PET_input[i,0])
        # print(mri_min, mri_max, pet_min, pet_max)
        
        plt.subplot(4,6,6*i+1)
        plt.axis('off')
        plt.imshow(MRI_full[i,0],cmap='gray', vmin=mri_min, vmax=mri_max)
        plt.colorbar()
        plt.title("Full MRI ("+subjects[i].split('/')[-3]+"_"+subjects[i].split('/')[-1]+")")
        
        plt.subplot(4,6,6*i+3)
        plt.axis('off')
        plt.imshow(anatomical_input[i,0],cmap='gray', vmin=mri_min, vmax=mri_max)
        plt.colorbar()
        plt.title("Anatomical input")
        
        plt.subplot(4,6,6*i+5)
        plt.axis('off')
        plt.imshow(PET_input[i,0], cmap='jet', vmin=pet_min, vmax=pet_max)
        plt.colorbar()
        plt.title("PET input")

        plt.subplot(4,6,6*i+2)
        plt.axis('off')
        plt.imshow(fusion[i,0], cmap='jet', vmin=pet_min, vmax=pet_max)
        plt.colorbar()
        plt.title("Fusion output")

        plt.subplot(4,6,6*i+4)
        plt.axis('off')
        plt.imshow(MRI_pred[i,0],cmap='gray', vmin=mri_min, vmax=mri_max)
        plt.colorbar()
        plt.title("MRI conversion output")
        
        plt.subplot(4,6,6*i+6)
        plt.axis('off')
        plt.imshow(PET_pred[i,0], cmap='jet', vmin=pet_min, vmax=pet_max)
        plt.colorbar()
        plt.title("PET conversion output")
    if state:    
        filename="train_epoch"+str(epoch)+"_"+str(iter)+".png"
    else:
        filename="valid_epoch"+str(epoch)+"_"+str(iter)+".png"
        
    filefolder=output_folder+"output_images/"

    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()

def record_parser(output_folder, args):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(os.path.join(output_folder,"command.txt"), 'w')
    f.write(str(args))
    f.close()

def record_search_config(config, output_folder):
    json_file_path = output_folder+"best_config.json"

    # Save the dictionary to a JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(config, json_file, indent=4)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, name="accuracy", minmax=0, output_folder=""
    ):
        self.best_epoch=0
        self.minmax = minmax
        if self.minmax==0:
            self.best_para = float(100)
        if self.minmax==1:
            self.best_para = float(0)
        self.name = name
        self.output_folder = output_folder
    def __call__(
        self, current_para, epoch, fold, model_x2y, model_y2x=None
    ):
        if not os.path.exists(self.output_folder+"checkpoint/"):
            os.makedirs(self.output_folder+"checkpoint/")
        if self.minmax==1:
            if current_para > self.best_para:
                self.best_para = current_para
                
                print("\n------------>Best parameter {} : {}".format(self.name, self.best_para))
                print("\nSaved at epoch: {}\n".format(epoch))

                model_out_path = self.output_folder+"/checkpoint/model_fusion_best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                torch.save(model_x2y.state_dict(), model_out_path)
                if model_y2x!=None:
                    model_out_path = self.output_folder+"/checkpoint/model_conversion_best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                    torch.save(model_y2x.state_dict(), model_out_path)
                self.best_epoch=epoch
        if self.minmax==0:
            if current_para < self.best_para:
                self.best_para = current_para

                print("\n------------>Best parameter {} : {}".format(self.name, self.best_para))
                print("\nSaved at epoch: {}\n".format(epoch))

                model_out_path = self.output_folder+"/checkpoint/model_fusion_best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                torch.save(model_x2y.state_dict(), model_out_path)
                if model_y2x!=None:
                    model_out_path = self.output_folder+"/checkpoint/model_conversion_best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                    torch.save(model_y2x.state_dict(), model_out_path)
                self.best_epoch=epoch

