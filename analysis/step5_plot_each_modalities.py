import numpy as np

import nibabel as nib


import matplotlib.pyplot as plt
import os 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np  
def plot_individuals(input_filepaths):

    img_list=[]
    for i in range(len(input_filepaths)):
        input_filepath=input_filepaths[i]
        if input_filepath.find("IY")==-1:
            title=input_filepath.split('/')[-1]
        else:
            title="PVC"

    
        CN_filepath=os.path.join(input_filepath, filename[0])
        CN_nii=np.array(nib.load(CN_filepath).get_fdata())

        axial=np.concatenate((np.zeros([16, 109]),CN_nii[7:-7, :, 35],np.zeros([16, 109])), axis=0)
        axial=np.concatenate((np.zeros([109, 7]),axial[:, 7:-7],np.zeros([109, 7])), axis=1)
        axial=np.swapaxes(axial, axis1=1, axis2=0)
        

        coronal=np.concatenate((np.zeros([91, 14]),CN_nii[:, 46, 5:-10],np.zeros([91, 19])), axis=1)
        coronal=np.concatenate((np.zeros([16, 109]),coronal[7:-7, :],np.zeros([16, 109])), axis=0)
        coronal=np.flip(coronal, axis=1)
        coronal=np.swapaxes(coronal, axis1=1, axis2=0)
        
        sagittal=np.concatenate((np.zeros([109, 9]),CN_nii[50, :, :-7],np.zeros([109, 16])), axis=1)
        sagittal=np.concatenate((np.zeros([16, 109]),sagittal[7:-7, :],np.zeros([16, 109])), axis=0)
        sagittal=np.flip(sagittal, axis=1)
        sagittal=np.swapaxes(sagittal, axis1=1, axis2=0)
        
        CN_array=np.concatenate([axial, sagittal, coronal], axis=1)
        
        MCI_filepath=os.path.join(input_filepath, filename[1])
        MCI_nii=np.array(nib.load(MCI_filepath).get_fdata())
        
            
        axial=np.concatenate((np.zeros([16, 109]),MCI_nii[7:-7, :, 35],np.zeros([16, 109])), axis=0)
        axial=np.concatenate((np.zeros([109, 7]),axial[:, 7:-7],np.zeros([109, 7])), axis=1)
        axial=np.swapaxes(axial, axis1=1, axis2=0)
        
        coronal=np.concatenate((np.zeros([91, 14]),MCI_nii[:, 46, 5:-10],np.zeros([91, 19])), axis=1)
        coronal=np.concatenate((np.zeros([16, 109]),coronal[7:-7, :],np.zeros([16, 109])), axis=0)
        coronal=np.flip(coronal, axis=1)
        coronal=np.swapaxes(coronal, axis1=1, axis2=0)
        
        sagittal=np.concatenate((np.zeros([109, 9]),MCI_nii[50, :, :-7],np.zeros([109, 16])), axis=1)
        sagittal=np.concatenate((np.zeros([16, 109]),sagittal[7:-7, :],np.zeros([16, 109])), axis=0)
        sagittal=np.flip(sagittal, axis=1)
        sagittal=np.swapaxes(sagittal, axis1=1, axis2=0)
        
        MCI_array=np.concatenate([axial, sagittal, coronal], axis=1)
        
        
        AD_filepath=os.path.join(input_filepath, filename[2])
        AD_nii=np.array(nib.load(AD_filepath).get_fdata())

        
        axial=np.concatenate((np.zeros([16, 109]),AD_nii[7:-7, :, 35],np.zeros([16, 109])), axis=0)
        axial=np.concatenate((np.zeros([109, 7]),axial[:, 7:-7],np.zeros([109, 7])), axis=1)
        axial=np.swapaxes(axial, axis1=1, axis2=0)
        
        coronal=np.concatenate((np.zeros([91, 14]),AD_nii[:, 46, 5:-12],np.zeros([91, 21])), axis=1)
        coronal=np.concatenate((np.zeros([16, 109]),coronal[7:-7, :],np.zeros([16, 109])), axis=0)
        coronal=np.flip(coronal, axis=1)
        coronal=np.swapaxes(coronal, axis1=1, axis2=0)
        
        sagittal=np.concatenate((np.zeros([109, 9]),AD_nii[50, :, :-11],np.zeros([109, 20])), axis=1)
        sagittal=np.concatenate((np.zeros([16, 109]),sagittal[7:-7, :],np.zeros([16, 109])), axis=0)
        sagittal=np.flip(sagittal, axis=1)
        sagittal=np.swapaxes(sagittal, axis1=1, axis2=0)
        
        AD_array=np.concatenate([axial, sagittal, coronal], axis=1)
        

        img_list.append([CN_array, MCI_array, AD_array])  


        if title.find("MRI") ==-1:
            img_min=0
            img_max=2
            cmap='jet'
            cbar_label="Tracer activity (SUVr)"
        else:
            img_min=0
            img_max=1
            cmap='gray'
            cbar_label="Tracer activity (a.u.)"


        plt.figure(figsize=(15,15))
        plt.title(title, fontsize=36)
        plt.subplot(3, 1, 1)
        plt.axis('off')
        img=plt.imshow(CN_array, cmap=cmap, vmin=img_min, vmax=img_max)
        plt.subplot(3, 1, 2)
        plt.axis('off')
        plt.imshow(MCI_array, cmap=cmap, vmin=img_min, vmax=img_max)
        plt.subplot(3, 1, 3)
        plt.axis('off')
        plt.imshow(AD_array, cmap=cmap, vmin=img_min, vmax=img_max)
        cax = plt.axes((0.88, 0.15, 0.02, 0.7))
        cbar=plt.colorbar(img, cax=cax)
        cbar.set_label(cbar_label, fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=20)

        plt.savefig("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/visualization/all_modalities/{}.png".format(title),bbox_inches='tight')
        plt.close()
    return img_list
        
    
def plot_differences(reference_img_list, reference_paths, config_img_list, config_paths):

    for ref in range(len(reference_img_list)):
        for config in range(len(config_img_list)):
            CN_array=config_img_list[config][0]-reference_img_list[ref][0]
            MCI_array=config_img_list[config][1]-reference_img_list[ref][1]
            AD_array=config_img_list[config][2]-reference_img_list[ref][2]

            
            title='Difference_map({}-{}).png'.format(config_paths[config].split('/')[-1], reference_paths[ref].split('/')[-1])
            img_max=2
            img_min=-2
            cmap="seismic"
            cbar_label="Tracer activity (SUVr)"

            plt.figure(figsize=(15,15))
            plt.title(title, fontsize=36)
            plt.subplot(3, 1, 1)
            plt.axis('off')
            img=plt.imshow(CN_array, cmap=cmap, vmin=img_min, vmax=img_max)
            plt.subplot(3, 1, 2)
            plt.axis('off')
            plt.imshow(MCI_array, cmap=cmap, vmin=img_min, vmax=img_max)
            plt.subplot(3, 1, 3)
            plt.axis('off')
            plt.imshow(AD_array, cmap=cmap, vmin=img_min, vmax=img_max)
            cax = plt.axes((0.88, 0.15, 0.02, 0.7))
            cbar=plt.colorbar(img, cax=cax)
            cbar.set_label(cbar_label, fontsize=20)
            cbar.ax.tick_params(axis='y', labelsize=20)

            plt.savefig("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/visualization/all_modalities/{}.png".format(title),bbox_inches='tight')
            plt.close()





if __name__ == "__main__":
    global filename
    filename=['CN/003_S_4441.nii', 'MCI/002_S_4171.nii', "AD/002_S_1268.nii"]
    
    # nii_filelist=['MRI.nii', 'PET.nii', 'fusion.nii', 'output_RBV_suvr.nii']
    reference_paths=['/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/MRI', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/PET', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY']#, 'label_3d_three_components.nii']
    
    # fig_title_list=['MRI', 'PET', 'PVC']
    # , 'Baseline', 'MRI-styled PET', 'Difference map (PVC - PET)', 'Difference map (Baseline - PET)', 'Difference map (Baseline - PVC)', 'Difference map (MRI-styled PET - PET)', 'Difference map (MRI-styled PET - PVC)']


    config_paths=['/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/Baseline/fold0/Baseline', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigA/fold0/ConfigA_real', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigA/fold0/ConfigA_pseudo', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigB/fold0/ConfigB_real', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigB/fold0/ConfigB_pseudo', "/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY", '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigC/fold0/ConfigC_real', '/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment/ConfigD/fold0/ConfigD_real']
    
    reference_img_list=plot_individuals(reference_paths)
    config_img_list=plot_individuals(config_paths)

    plot_differences(reference_img_list[1:], reference_paths[1:], config_img_list, config_paths)


    
    
    
    
        
    # for i in range(len(fig_title_list)):   
        
    #     if i==0:
    #         CN_array=img_list[i][0]
    #         MCI_array=img_list[i][1]
    #         AD_array=img_list[i][2]
            
            
    #         img_min=0
    #         img_max=1
    #         plt.figure(figsize=(15,15))
    #         plt.subplot(3, 1, 1)
    #         # plt.title(fig_title_list[i], fontsize=36)
    #         plt.axis('off')
    #         img=plt.imshow(CN_array, cmap='gray', vmin=img_min, vmax=img_max)
            
            
    #         plt.subplot(3, 1, 2)
    #         plt.axis('off')
    #         plt.imshow(MCI_array, cmap='gray', vmin=img_min, vmax=img_max)
            
    #         plt.subplot(3, 1, 3)
    #         plt.axis('off')
    #         plt.imshow(AD_array, cmap='gray', vmin=img_min, vmax=img_max)
            
    #         cax = plt.axes((0.88, 0.15, 0.02, 0.7))
    #         cbar=plt.colorbar(img, cax=cax)
    #         cbar.set_label("Tracer activity (a.u.)", fontsize=20)
            
    #         cbar.ax.tick_params(axis='y', labelsize=20)
                
    #         # plt.tight_layout()
    #         plt.savefig("/home/linyunong/project/style_transfer/analysis/visualization/all_modalities/{}.png".format(fig_title_list[i]),bbox_inches='tight')
    #         plt.close()
            
            
            
    #     elif i<5:
    #         CN_array=img_list[i][0]
    #         MCI_array=img_list[i][1]
    #         AD_array=img_list[i][2]
            
    #         img_min=0
    #         img_max=2
    #         plt.figure(figsize=(15,15))
            
    #         plt.subplot(3, 1, 1)
    #         # plt.title(fig_title_list[i], fontsize=36)
    #         plt.axis('off')
    #         img=plt.imshow(CN_array, cmap='jet', vmin=img_min, vmax=img_max)
            
    #         plt.subplot(3, 1, 2)
    #         plt.axis('off')
    #         plt.imshow(MCI_array, cmap='jet', vmin=img_min, vmax=img_max)
            
    #         plt.subplot(3, 1, 3)
    #         plt.axis('off')
    #         plt.imshow(AD_array, cmap='jet', vmin=img_min, vmax=img_max)
            
            
    #         cax = plt.axes((0.88, 0.15, 0.02, 0.7))
    #         cbar=plt.colorbar(img, cax=cax)
    #         cbar.set_label("Tracer activity (SUVr)", fontsize=20)
    #         cbar.ax.tick_params(axis='y', labelsize=20)
            
    #         # plt.tight_layout()
    #         plt.savefig("/home/linyunong/project/style_transfer/analysis/visualization/all_modalities/{}.png".format(fig_title_list[i]),bbox_inches='tight')
    #         plt.close()
            
            
    #         # if i==1:
    #         #     cax_color_pet = plt.axes((0.63, 0.52, 0.01, 0.36))
    #         #     cbar_color_pet=plt.colorbar(img_color_pet, cax=cax_color_pet)
    #         # else:  
    #         #     cax_color_pvc = plt.axes((0.91, 0.52, 0.01, 0.36))
    #         #     cbar_color_pvc=plt.colorbar(img_color_pvc, cax=cax_color_pvc)
        
            
    #     else:
            
    #         if i==5:
    #             img_min=-1
    #             img_max=1
                
    #             CN_array=img_list[3][0]-img_list[1][0]
    #             MCI_array=img_list[3][1]-img_list[1][1]
    #             AD_array=img_list[3][2]-img_list[1][2]
                
    #         elif i==6:
                            
    #             img_min=-5
    #             img_max=5
    #             CN_array=img_list[3][0]-img_list[2][0]
    #             MCI_array=img_list[3][1]-img_list[2][1]
    #             AD_array=img_list[3][2]-img_list[2][2]
            
    #         elif i==7:
                
    #             img_min=-3
    #             img_max=3
    #             CN_array=img_list[2][0]-img_list[1][0]
    #             MCI_array=img_list[2][1]-img_list[1][1]
    #             AD_array=img_list[2][2]-img_list[1][2]
                
    #         elif i==8:
    #             img_min=-1
    #             img_max=1
                
    #             CN_array=img_list[4][0]-img_list[1][0]
    #             MCI_array=img_list[4][1]-img_list[1][1]
    #             AD_array=img_list[4][2]-img_list[1][2]
                
    #         elif i==9:
                            
    #             img_min=-1
    #             img_max=1
    #             CN_array=img_list[3][0]-img_list[4][0]
    #             MCI_array=img_list[3][1]-img_list[4][1]
    #             AD_array=img_list[3][2]-img_list[4][2]
            
    #         elif i==10:
                
    #             img_min=-5
    #             img_max=5
    #             CN_array=img_list[4][0]-img_list[2][0]
    #             MCI_array=img_list[4][1]-img_list[2][1]
    #             AD_array=img_list[4][2]-img_list[2][2]
                
    #         elif i==11:
            
    #             img_min=0
    #             img_max=1
                
    #             CN_array=img_list[5][0]
    #             MCI_array=img_list[5][1]
    #             AD_array=img_list[5][2]

    #         plt.figure(figsize=(15,15))

    #         plt.subplot(3, 1, 1)
    #         plt.axis('off')
    #         # plt.title(fig_title_list[i], fontsize=36)
    #         if i==11:
    #             img=plt.imshow(CN_array, vmin=img_min, vmax=img_max, cmap='gray')
    #         else:
    #             img=plt.imshow(CN_array, cmap='seismic', vmin=img_min, vmax=img_max)
            
    #         plt.subplot(3, 1, 2)
    #         plt.axis('off')
    #         if i==11:
    #             plt.imshow(MCI_array, vmin=img_min, vmax=img_max, cmap='gray')
    #         else:
    #             plt.imshow(MCI_array, cmap='seismic', vmin=img_min, vmax=img_max)    
            
    #         plt.subplot(3, 1, 3)
    #         plt.axis('off')
    #         if i==11:
    #             plt.imshow(AD_array, vmin=img_min, vmax=img_max, cmap='gray')
    #         else:
    #             plt.imshow(AD_array, cmap='seismic', vmin=img_min, vmax=img_max)
                
        
    #         cax = plt.axes((0.88, 0.15, 0.02, 0.7))
    #         cbar=plt.colorbar(img, cax=cax)
            
    #         cbar.set_label("Tracer activity (SUVr)", fontsize=20)

    #         cbar.ax.tick_params(axis='y', labelsize=20)
            
    #         # plt.tight_layout()
            
    #         plt.savefig("/home/linyunong/project/style_transfer/analysis/visualization/all_modalities/{}.png".format(fig_title_list[i]),bbox_inches='tight')   
    #         plt.close()
            
    
    