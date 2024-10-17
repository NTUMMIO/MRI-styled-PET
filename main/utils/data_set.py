from hashlib import new
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import torchvision.transforms as trns
import matplotlib.pyplot as plt
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
    
# class TriDataset(Dataset):
#     def __init__(self, root0, root1, root2, root3, gt, transform):

#         self.transform = transform
#         self.root0=root0
#         self.root1=root1
#         self.root2=root2
#         self.root3=root3
#         self.gt=gt
        

#     def __getitem__(self, index):

#         imgpath0 = self.root0[index]
#         img0 = np.load("/home/linyunong/project/"+imgpath0)
#         img0=img0.astype(np.float32)
         
#         imgpath1 = self.root1[index]
#         img1 = np.load("/home/linyunong/project/"+imgpath1)
#         img1= img1.astype(np.float32)
        
#         imgpath2 = self.root2[index]
#         img2 = np.load("/home/linyunong/project/"+imgpath2)
#         img2= img2.astype(np.float32)
        
#         imgpath3 = self.root3[index]
#         img3 = np.load("/home/linyunong/project/"+imgpath3)
#         img3= img3.astype(np.float32)
        
#         # print(self.gt.shape)
#         gt=self.gt[index]
        
#         # print(img0.shape, img1.shape, img2.shape, img3.shape)
#         o=np.concatenate([img0[None,:3,:,:],img1[None,:3,:,:],img2[None,:3,:,:],img3[None,:3,:,:]],axis=0)
#         # print('o.shape',o.shape)
#         # print(o)
#         return o, imgpath0, gt

#     def __len__(self):
#         return len(self.root0) 

class TriDataset(Dataset):
    def __init__(self, root0, root1, transform):

        self.transform = transform
        self.root0=root0
        self.root1=root1
        
    def __getitem__(self, index):

        imgpath0 = self.root0[index]
        img0 = np.load(imgpath0)#"/home/linyunong/project/"+
        img0=img0.astype(np.float32)
         
        imgpath1 = self.root1[index]
        img1 = np.load(imgpath1)#"/home/linyunong/project/"+
        img1= img1.astype(np.float32)
   
        return img0, img1, imgpath0, imgpath1
    

    def __len__(self):
        return len(self.root0) 
    
class ADNIDataset(Dataset):
    def __init__(self, root0, root1, inputScale, transform):

        self.transform = transform
        self.root0=root0
        self.root1=root1
        self.inputScale = inputScale
        
    def __getitem__(self, index):

        imgpath0 = self.root0[index]
        img0 = np.load(imgpath0)#"/home/linyunong/project/"+
        img0=img0.astype(np.float32)
        if self.inputScale:
            img0=zoom(img0, zoom=[1, 2, 2], order=1)
            # img0_slices=[]
            # for c in range(img0.shape[0]):
            #     slices=zoom(img0[c, :, :], zoom=2, order=1)
            #     img0_slices.append(slices)
            # img0_slices=np.array(img0_slices)
            # img0=img0_slices

         
        imgpath1 = self.root1[index]
        img1 = np.load(imgpath1)#"/home/linyunong/project/"+
        img1= img1.astype(np.float32)

        if self.inputScale:
            # img1_slices=[]
            # for c in range(img1.shape[0]):
            #     slices=zoom(img1[c, :, :], zoom=2, order=1)
            #     img1_slices.append(slices)
            # img1_slices=np.array(img1_slices)
            # img1=img1_slices
            img1=zoom(img1, zoom=[1, 2, 2], order=1)

   
        return img0, img1, imgpath0, imgpath1
    

    def __len__(self):
        return len(self.root0) 
    

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


class MinMaxScaler:
    def __init__(self):
        self.pet_min = None
        self.pet_max = None
        self.mr_min = None
        self.mr_max = None

    def fit(self, pet, mr):
        """
        Compute the min and max values for each feature (channel).
        Args:
        - data: A 4D tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels, H and W are the height and width.
        """
        # Compute the minimum and maximum values per channel (axis 0)
        self.pet_min = np.min(pet, axis=(0, 1, 2), keepdims=True) # axis=(0, 2, 3)
        self.pet_max = np.max(pet, axis=(0, 1, 2), keepdims=True) # axis=(0, 2, 3)
        # print("pet min max", pet.shape, self.pet_min, self.pet_max)

        self.mr_min = np.min(mr, axis=(0, 1, 2), keepdims=True) # axis=(0, 2, 3)
        self.mr_max = np.max(mr, axis=(0, 1, 2), keepdims=True) # axis=(0, 2, 3)
        # save the min and max

    def transform(self, pet, mr):
        """
        Apply min-max scaling to transform the data to range [0, 1].
        Args:
        - data: A 4D tensor of shape (N, C, H, W).
        
        Returns:
        - Scaled data with values between 0 and 1.
        """
        if self.pet_min is None or self.pet_max is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'transform'.")
        
        # Apply min-max scaling
        scaled_pet = (pet - self.pet_min) / (self.pet_max - self.pet_min + 1e-5)  # Add epsilon to avoid division by zero
        
        
        if self.mr_min is None or self.mr_max is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'transform'.")
        
        # Apply min-max scaling
        scaled_mr = (mr - self.mr_min) / (self.mr_max - self.mr_min + 1e-5)  # Add epsilon to avoid division by zero
        
        return scaled_pet, scaled_mr

    def inverse_transform(self, scaled_pet, scaled_mr, pet_min, pet_max, mr_min, mr_max):
        """
        Reverse the min-max scaling to recover the original data range.
        Args:
        - scaled_data: A 4D tensor of shape (N, C, H, W) with values between 0 and 1.
        
        Returns:
        - Original data scaled back to the original range using the stored min and max values.
        """
        if pet_min is None or pet_max is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'inverse_transform'.")
        
        # Reverse the scaling transformation
        original_pet = scaled_pet * (pet_max - pet_min + 1e-5) + pet_min
        
        if mr_min is None or mr_max is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'inverse_transform'.")
        
        # Reverse the scaling transformation
        original_mr = scaled_mr * (mr_max - mr_min + 1e-5) + mr_min
        
        return original_pet, original_mr
    def get_min_max(self):
        return self.pet_min, self.pet_max, self.mr_min, self.mr_max

# Example usage
# N, C, H, W = 8, 1, 112, 112  # Example batch size, channels, height, and width
# data = torch.rand((N, C, H, W))  # Random input tensor

# scaler = MinMaxScaler()
# scaler.fit(data)  # Fit the scaler to compute min and max values
# scaled_data = scaler.transform(data)  # Apply the min-max scaling
# original_data = scaler.inverse_transform(scaled_data)  # Reverse the scaling to get the original data back

# print("Original Data Min:", data.min().item())
# print("Scaled Data Min:", scaled_data.min().item())
# print("Recovered Data Min:", original_data.min().item())


class InferenceDataset(Dataset):
    def __init__(self, input_directory, suvr, segmentation, simulation, transform):
        self.input_directory = input_directory
        self.suvr = suvr
        self.segmentation = segmentation
        self.simulation = simulation
        self.transform = transform

        # self.subject_filenames: all filenames under src/input/native_space/
        self.subject_filenames=os.listdir(self.input_directory)
        # print(self.subject_filenames)
        # if self.suvr:
        #     self.subject_filenames.remove("AD_024_S_4905")
        # else: 
        #     self.subject_filenames.remove("AD_027_S_4964")

        # if self.segmentation==3:
        #     self.subject_filenames=["AD_027_S_4801", "AD_027_S_4964", "CN_036_S_4389", "AD_098_S_0160", "AD_098_S_0269", "AD_141_S_4426", "AD_305_S_6810"]

        self.scaler = MinMaxScaler()
        
        # self.norm_matrix = {}
        
    #     # Load raw nifti file
    #     for subject_filename in subject_filenames:

    #     
    # scaler.fit(data)  # Fit the scaler to compute min and max values
    # scaled_data = scaler.transform(data)



        
    def __getitem__(self, index):

        subject_filename = self.subject_filenames[index]
        print(subject_filename)
        if self.simulation==0:
            if self.suvr:
                if os.path.isfile(os.path.join(self.input_directory, subject_filename, "SUVr.nii")):
                    pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "SUVr.nii")).get_fdata()#[:, :, :, 0].astype(np.float32)#
                else:
                    pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "PET.nii")).get_fdata()
            else:
                if os.path.isfile(os.path.join(self.input_directory, subject_filename, "SUV.nii")):
                    pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "SUV.nii")).get_fdata()#[:, :, :, 0].astype(np.float32)
                else:
                    pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "PET.nii")).get_fdata()
                    # print("--", np.min(pet_input), np.max(pet_input))
        elif self.simulation==1:
            pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "PET_FS.nii")).get_fdata()
            # print("___________________", np.max(pet_input), np.min(pet_input))
        elif self.simulation==2:
            pet_input=nib.load(os.path.join(self.input_directory, subject_filename, "PET_SPM.nii")).get_fdata()
        

        mr_input=nib.load(os.path.join(self.input_directory, subject_filename, "MR.nii")).get_fdata()#[:, :, :, 0]#.astype(np.float32)#
        if os.path.isfile(os.path.join(self.input_directory, subject_filename, "aseg.nii")):
            label=nib.load(os.path.join(self.input_directory, subject_filename, "aseg.nii")).get_fdata()
        else:
            label=nib.load(os.path.join(self.input_directory, subject_filename, "MR.nii")).get_fdata()

        if len(pet_input.shape)==4:
            pet_input=pet_input[:, :, :, 0]
        if len(mr_input.shape)==4:
            mr_input=mr_input[:, :, :, 0]
        pet_input=np.nan_to_num(pet_input)
        mr_input=np.nan_to_num(mr_input) 
        label=np.nan_to_num(label) 
        

        # print(subject_filename)
        # print(pet_input.shape)
        # print(mr_input.shape)
        # print(label.shape)
        pet_input=np.swapaxes(pet_input, 0, 2)
        pet_input=np.swapaxes(pet_input, 1, 2)
        mr_input=np.swapaxes(mr_input, 0, 2)
        mr_input=np.swapaxes(mr_input, 1, 2)
        label=np.swapaxes(label, 0, 2)
        label=np.swapaxes(label, 1, 2)
        
        
        
        # pet_min, pet_max, mr_min, mr_max = self.scaler.get_min_max()

        # self.norm_matrix[subject_filename] = np.array(self.scaler.get_min_max())


        #===============================================
        # if SPM segmetation exists, extract brain tissue only
        #===============================================
        if os.path.isfile(os.path.join(self.input_directory, subject_filename, "c1MR.nii")) and os.path.isfile(os.path.join(self.input_directory, subject_filename, "c2MR.nii")) and os.path.isfile(os.path.join(self.input_directory, subject_filename, "c3MR.nii")):

            brain_mask=nib.load(os.path.join(self.input_directory, subject_filename, "c1MR.nii")).get_fdata()+nib.load(os.path.join(self.input_directory, subject_filename, "c2MR.nii")).get_fdata()+nib.load(os.path.join(self.input_directory, subject_filename, "c3MR.nii")).get_fdata() 
            brain_mask=np.swapaxes(brain_mask, 0, 2)
            brain_mask=np.swapaxes(brain_mask, 1, 2)

            brain_mask = np.where(brain_mask>0.7, 1, 0).astype(mr_input.dtype)

            pet_input=pet_input*brain_mask
            mr_input=mr_input*brain_mask
            
            brain_mask = brain_mask[:, None, :, :].astype(np.float32)
        else:
            brain_mask=0
            
        
       
        if self.segmentation==3:
            gm_mask=nib.load(os.path.join(self.input_directory, subject_filename, "c1MR.nii")).get_fdata()
            wm_mask=nib.load(os.path.join(self.input_directory, subject_filename, "c2MR.nii")).get_fdata()

            gm_mask=np.swapaxes(gm_mask, 0, 2)
            gm_mask=np.swapaxes(gm_mask, 1, 2)

            wm_mask=np.swapaxes(wm_mask, 0, 2)
            wm_mask=np.swapaxes(wm_mask, 1, 2)

            gm_mask = np.where(gm_mask>0.7, 1, 0).astype(mr_input.dtype)
            wm_mask = np.where(wm_mask>0.7, 0.25, 0).astype(mr_input.dtype)

            spm_mask = gm_mask + wm_mask
            spm_mask = spm_mask[:, None, :, :].astype(np.float32)
        else:
            spm_mask=0


            

        # print(pet_input.shape)
        # print(mr_input.shape)
        # print(label.shape)
        # print(brain_mask.shape)

        
        
        
        self.scaler.fit(pet_input, mr_input)
        pet_scaled, mr_scaled = self.scaler.transform(pet_input, mr_input)
        # else:
            # print("no spm")
            # brain_mask=np.ones(pet_scaled.shape).astype(np.float32)
        
        return pet_scaled[:, None, :, :].astype(np.float32), mr_scaled[:, None, :, :].astype(np.float32), label[:, None, :, :].astype(np.float32), subject_filename, np.array(self.scaler.get_min_max()).astype(np.float32), brain_mask, spm_mask
        
    

    def __len__(self):
        return len(self.subject_filenames) 
    
# class ConversionDataset(Dataset):
#     def __init__(self, root0, root1, root2, root3, gt, transform):

#         self.transform = transform
#         self.root0=root0
#         self.root1=root1
#         self.root2=root2
#         self.root3=root3
#         self.gt=gt
        

#     def __getitem__(self, index):

#         imgpath0 = self.root0[index]
#         img0 = np.load("/home/linyunong/project/"+imgpath0)
#         img0=img0.astype(np.float32)[0]
#         # print(img0.shape)
         
#         imgpath1 = self.root1[index]
#         img1 = np.load("/home/linyunong/project/"+imgpath1)
#         img1= img1.astype(np.float32)[0]
#         # print(img1.shape)
        
#         imgpath2 = self.root2[index]
#         img2 = np.load(imgpath2)
#         img2= img2.astype(np.float32)
#         # print(img2.shape)
        
#         imgpath3 = self.root3[index]
#         img3 = np.load(imgpath3)
#         img3= img3.astype(np.float32)
#         # print(img3.shape)
        
#         gt=self.gt[index]
        
#         # print(img0.shape, img1.shape, img2.shape, img3.shape)
#         o=np.concatenate([img0[None,None,:,:],img1[None,None,:,:],img2[None,None,:,:],img3[None,None,:,:]],axis=0)
#         # print('o.shape',o.shape)
#         # print(o)
#         return o, imgpath0, gt

#     def __len__(self):
#         return len(self.root0) 


if __name__ == '__main__':


    MRI_valid=[]
    FDG_valid=[] 
    MRI_inverted_valid=[]
    

    f_name='src/fold/fold0/valid_MRI.txt'
    f_name=f_name.replace('fold/', 'fold_112_slice_norm/')
    f = open(f_name, 'r')  
    for line in f.readlines():
        MRI_valid.append(str(line)[:-2])
        print(str(line)[:-2])
        
    f_name='src/fold/fold0/valid_PET.txt'
    f_name=f_name.replace('fold/', 'fold_112_slice_norm/')
    f = open(f_name, 'r')
    for line in f.readlines():
        FDG_valid.append(str(line)[:-2])
        
    f_name='src/fold/fold0/valid_MRI.txt'
    f = open(f_name, 'r')  
    for line in f.readlines():
        MRI_inverted_valid.append(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
        print(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))

    valid_dataset = TriDataset(root0=FDG_valid,root1=MRI_valid,root2=MRI_inverted_valid,transform=trns.Compose([trns.ToTensor()]))
    print(valid_dataset[0][1])
        
    
    
    