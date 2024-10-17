import os
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import pandas as pd

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def calc_psnr_and_ssim_slice_average(nii_source, nii_target):
    num_slices = nii_source.shape[-1]
    psnr_values = []
    ssim_values = []

    # nii_source=(nii_source-np.min(nii_source))/(np.max(nii_source)-np.min(nii_source)) 
    # nii_target=(nii_target-np.min(nii_target))/(np.max(nii_target)-np.min(nii_target)) 
    

    for slice_idx in range(num_slices):
        # Extract slices
        slice_source = nii_source[..., slice_idx]
        slice_target = nii_target[..., slice_idx]

        slice_source=(slice_source-np.min(slice_source))/(np.max(slice_source)-np.min(slice_source)) 
        slice_target=(slice_target-np.min(slice_target))/(np.max(slice_target)-np.min(slice_target)) 
        # print(np.max(slice_source), np.min(slice_target))
        # Compute PSNR
        psnr = peak_signal_noise_ratio(slice_source, slice_target, data_range=1.0)
        # print(psnr)

        # Compute SSIM
        ssim = structural_similarity(slice_source, slice_target, data_range=1.0)

        if ~np.isnan(psnr) and ~np.isnan(ssim) and ~np.isinf(psnr) and ~np.isinf(ssim):
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        # print(ssim)

    # print(sum(psnr_values) / len(psnr_values))
    # print(sum(ssim_values) / len(ssim_values))
    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)
def extract_brain(image, group, subject_name):
    # ==========================================
    # Apply binary mask to keep only the brain tissue (excluding soft tissue, bones and other tissues)
    # ==========================================
    brain_mask_folder="/home/linyunong/project/src/ADNI-MRI-FDG/{}_MRI_skull_strip_mask/{}".format(group, subject_name)
    soft_tissue=nib.load(os.path.join(brain_mask_folder, 'c4{}'.format(subject_name))).get_fdata()
    bones=nib.load(os.path.join(brain_mask_folder, 'c5{}'.format(subject_name))).get_fdata()
    other_tissue=nib.load(os.path.join(brain_mask_folder, 'c6{}'.format(subject_name))).get_fdata()
    brain_tissue_binary_mask=soft_tissue+bones+other_tissue
    brain_tissue_binary_mask=np.where(brain_tissue_binary_mask > 0.5, 0, 1)

    image=image*brain_tissue_binary_mask
    return image

if __name__=='__main__':                
    
    fusion_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment"

    pet_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/PET"
    mri_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/MRI"
    pvc_root_folder="/home/linyunong/project/pvc/src/output"
    # pvc_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY_8mm"

    # pet_root_folder="/home/linyunong/project/pvc/src/output"
    # mri_root_folder="/home/linyunong/project/pvc/src/output"
    # pvc_root_folder="/home/linyunong/project/pvc/src/output"

    csv_root_folder='/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/raw_quantitative_metrics'

    # config_folders=os.listdir(fusion_root_folder)
    config_folders=['ConfigC', 'ConfigD']#'ConfigB_ratio1', 'ConfigC_ratio1', 'ConfigD_ratio1']#'Baseline', 'ConfigA', 'ConfigB_ratio05', 'ConfigD_ratio05_woutBoundary'
    for config_folder in config_folders:
        fold_folders=os.listdir(os.path.join(fusion_root_folder, config_folder))
        for fold_folder in fold_folders:
            experiment_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder))
            print(experiment_folders)
            for experiment_folder in experiment_folders:
                # if experiment_folder.find('9-16')==-1 or experiment_folder.find('9-17')==-1:
                # if experiment_folder.find('train')!=-1:
                    
                    group_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder))
                    for group_folder in group_folders:
                        subject_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder, group_folder))
                        if subject_folders.count("016_S_4952.nii")> 0:
                            print("remove")
                            subject_folders.remove("016_S_4952.nii")
                        print(os.path.join(csv_root_folder, config_folder, fold_folder, experiment_folder, group_folder+'.csv'))
                        # df=pd.DataFrame(columns=['Subject', 'SSIM_PET', 'SSIM_PVC', 'PSNR_PET', 'PSNR_PVC'])
                        df=pd.DataFrame(columns=['Subject', 'SSIM_PET', 'SSIM_PVC', 'SSIM_MRI', 'PSNR_PET', 'PSNR_PVC', 'PSNR_MRI'])
                        for subject_folder in subject_folders:
                            
                            fusion_current_subject_path=os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder, group_folder, subject_folder)
                            pet_current_subject_path=os.path.join(pet_root_folder, group_folder, subject_folder)#[:-4], 'PET.nii')
                            mri_current_subject_path=os.path.join(mri_root_folder, group_folder, subject_folder)#[:-4], 'MRI.nii')
                            pvc_current_subject_path=os.path.join(pvc_root_folder, group_folder, subject_folder[:-4], 'output_IY_8mm_suvr.nii')

                            fusion=extract_brain(nib.load(fusion_current_subject_path).get_fdata(), group_folder, subject_folder)
                            pet=extract_brain(nib.load(pet_current_subject_path).get_fdata(), group_folder, subject_folder)
                            mri=extract_brain(nib.load(mri_current_subject_path).get_fdata(), group_folder, subject_folder)
                            pvc=extract_brain(nib.load(pvc_current_subject_path).get_fdata(), group_folder, subject_folder)

                            

                            # print(fusion.shape, pet.shape, pvc.shape)
                            # print(np.max(fusion), np.min(fusion))
                            # print(np.max(pet), np.min(pet))
                            # print(np.max(pvc), np.min(pvc))

                            # pet_psnr=peak_signal_noise_ratio(fusion, pet, data_range=np.max([fusion.max(), pet.max()]))#np.max(pet)-np.min(pet))
                            # pvc_psnr=peak_signal_noise_ratio(fusion, pvc, data_range=np.max([fusion.max(), pet.max()]))#np.max(pvc)-np.min(pvc))
                            # pet_ssim=structural_similarity(fusion, pet, data_range=np.max([fusion.max(), pet.max()]))#np.max(pet)-np.min(pet))
                            # pvc_ssim=structural_similarity(fusion, pvc, data_range=np.max([fusion.max(), pet.max()]))#np.max(pvc)-np.min(pvc))

                            pet_psnr, pet_ssim=calc_psnr_and_ssim_slice_average(pet, fusion)
                            mri_psnr, mri_ssim=calc_psnr_and_ssim_slice_average(mri, fusion)
                            pvc_psnr, pvc_ssim=calc_psnr_and_ssim_slice_average(pvc, fusion)


                            row=[subject_folder, pet_ssim, pvc_ssim, mri_ssim, pet_psnr, pvc_psnr, mri_psnr]

                            # df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

                            df.loc[len(df)]=row
                            
                        if not os.path.exists(os.path.join(csv_root_folder, config_folder, fold_folder, experiment_folder)):
                            os.makedirs(os.path.join(csv_root_folder, config_folder, fold_folder, experiment_folder))
                        df.to_csv(os.path.join(csv_root_folder, config_folder, fold_folder, experiment_folder, group_folder+'.csv'))


                        






                        



