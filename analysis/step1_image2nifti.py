
import os
import numpy as np
from PIL import Image
import nibabel as nib
import shutil

def stack_and_normalize(input_path, output_path, group, subject_name):
    print(input_path)
    reference_cerebellum=nib.load("whole_cb_mask_CGMH.nii").get_fdata()[:, :, :, 0]
    brain_mask_folder="/home/linyunong/project/src/ADNI-MRI-FDG/{}_MRI_skull_strip_mask/{}.nii".format(group, subject_name)

    # Axial slices 
    axial_slices=[]
    for i in range(91):
        current_filename=subject_name+"_Axial_"+str(i)+".tif"
        current_slice=np.array(Image.open(os.path.join(input_path, current_filename)))
        current_slice=current_slice[1:-2]
        current_slice=current_slice[:,10:-11]
        axial_slices.append(current_slice)
        # print(current_slice.shape)
    axial_slices=np.array(axial_slices)
    axial_slices=np.swapaxes(axial_slices, 0, 2)

    # Coronal slices 
    coronal_slices=[]
    for i in range(109):
        current_filename=subject_name+"_Coronal_"+str(i)+".tif"
        current_slice=np.array(Image.open(os.path.join(input_path, current_filename)))
        current_slice=current_slice[10:-11]
        current_slice=current_slice[:,10:-11]
        current_slice=np.flip(current_slice,axis=0)

        coronal_slices.append(current_slice)
        # print(current_slice.shape)
    coronal_slices=np.array(coronal_slices)
    coronal_slices=np.swapaxes(coronal_slices, 0, 1)
    coronal_slices=np.swapaxes(coronal_slices, 0, 2)

    # Sagittal slices 
    sagittal_slices=[]
    for i in range(91):
        current_filename=subject_name+"_Sagittal_"+str(i)+".tif"
        current_slice=np.array(Image.open(os.path.join(input_path, current_filename)))
        current_slice=current_slice[:,1:-2]
        current_slice=current_slice[10:-11]
        current_slice=np.flip(current_slice,axis=0)

        sagittal_slices.append(current_slice)
        # print(current_slice.shape)
    sagittal_slices=np.array(sagittal_slices)
    sagittal_slices=np.swapaxes(sagittal_slices, 1, 2)
    # sagittal_slices=np.swapaxes(coronal_slices, 0, 2)

    slices=(axial_slices+coronal_slices+sagittal_slices)/3


    
    
    # nib.save(nib.Nifti1Image(slices, affine=nib.load("whole_cb_mask_CGMH.nii").affine), output_path)

    # ==========================================
    # Apply binary mask to keep only the brain tissue (excluding soft tissue, bones and other tissues)
    # ==========================================
    
    soft_tissue=nib.load(os.path.join(brain_mask_folder, 'c4{}.nii'.format(subject_name))).get_fdata()
    bones=nib.load(os.path.join(brain_mask_folder, 'c5{}.nii'.format(subject_name))).get_fdata()
    other_tissue=nib.load(os.path.join(brain_mask_folder, 'c6{}.nii'.format(subject_name))).get_fdata()
    brain_tissue_binary_mask=soft_tissue+bones+other_tissue
    brain_tissue_binary_mask=np.where(brain_tissue_binary_mask > 0.5, 0, 1)

    slices=slices*brain_tissue_binary_mask

    # ==========================================
    # Calculate reference region activity
    # ==========================================

    # reference_mean_activity=np.sum(slices*reference_cerebellum)/np.sum(reference_cerebellum)
    # slices=slices/reference_mean_activity

    # ==========================================
    # Divide the whole image with mean reference region activity
    # ==========================================

    
    
    print("pre", np.max(slices), np.min(slices))
    original_pet=nib.load("/home/linyunong/project/src/ADNI-MRI-FDG/"+group+"_FDG/"+subject_name+".nii").get_fdata()[:, :, :, 0]
    pet_min=np.min(original_pet, keepdims=True)
    pet_max=np.max(original_pet, keepdims=True)
    print("scale", pet_max, pet_min)
    slices = slices * (pet_max - pet_min + 1e-5) + pet_min
    print("post", np.max(slices), np.min(slices))
    
    
    nib.save(nib.Nifti1Image(slices, affine=nib.load("whole_cb_mask_CGMH.nii").affine), output_path)

    # return slices

def normalize(file_path):

    image=np.array(nib.load(file_path).get_fdata())

    reference_cerebellum=nib.load("whole_cb_mask_CGMH.nii").get_fdata()[:, :, :, 0]

    # ==========================================
    # Calculate reference region activity
    # ==========================================

    reference_mean_activity=np.sum(image*reference_cerebellum)/np.sum(reference_cerebellum)
    image=image/reference_mean_activity

    # ==========================================
    # Divide the whole image with mean reference region activity
    # ==========================================

    nib.save(nib.Nifti1Image(image, affine=nib.load("whole_cb_mask_CGMH.nii").affine), file_path)

    # return image



def copy_and_rename(src_path, dest_path, new_name):
    # Copy the file
    shutil.copy(src_path, dest_path)

    # Rename the copied file
    new_path = f"{dest_path}/{new_name}"
    shutil.move(f"{dest_path}/{src_path}", new_path)


if __name__=='__main__':                
    
    fusion_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_images"

    fusion_target_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion"

    source_root_folder="/home/linyunong/project/src/ADNI-MRI-FDG" # AD_FDG
    mri_root_folder="/home/linyunong/project/src/ADNI-MRI-FDG" # AD_MRI_skull_strip

    pvc_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY"

    # config_folders=['ConfigA', 'ConfigB_ratio1', 'ConfigC_ratio1', 'ConfigD_ratio1']
    config_folders=['anat']#, 'ConfigB_ratio05', 'ConfigD_ratio05_woutBoundary']


    for config_folder in config_folders:
        fold_folders=os.listdir(os.path.join(fusion_root_folder, config_folder))
        for fold_folder in fold_folders:
            experiment_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder))

            for experiment_folder in experiment_folders:
                # if experiment_folder.find('9-12')==-1 and experiment_folder.find('9-13')==-1:
                # if experiment_folder.find('9-16')==-1 or experiment_folder.find('9-17')==-1:
                # if experiment_folder.find('9-19')!=-1 and experiment_folder.find('inference')==-1:#experiment_folder.find('2024-9-16-191956')!=-1 or experiment_folder.find('2024-9-18-13836')!=-1:

                    print(experiment_folder)
                    
                    print(os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder, 'checkpoint', 'model_fusion_best_loss_{}.pth'.format(fold_folder[:-1]+"_"+fold_folder[-1]), 'fusion_output'))
                    group_folders=os.listdir(os.path.join(fusion_root_folder, config_folder))
                    
                    for group_folder in group_folders:
                        subject_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, group_folder))
                        for subject_folder in subject_folders:

                            current_subject_input_path=os.path.join(fusion_root_folder, config_folder, group_folder, subject_folder)
                            
                            current_subject_output_path=os.path.join(fusion_target_folder, config_folder, group_folder, subject_folder)

                            # Check whether output folder existed
                            if not os.path.exists(os.path.join(fusion_target_folder, config_folder, group_folder)):
                                os.makedirs(os.path.join(fusion_target_folder, config_folder, group_folder))
                            # Check whether output file existed
                            if not os.path.isfile(current_subject_output_path):
                                current_subject_3darray=stack_and_normalize(current_subject_input_path, current_subject_output_path, group_folder, subject_folder.split('.')[0])

                            # Check whether MRI and PET already existed under the output_nifti folders
                            # if not os.path.exists(os.path.join(fusion_target_folder, 'source', 'PET', group_folder)):
                            #     os.makedirs(os.path.join(fusion_target_folder, 'source', 'PET', group_folder))
                            # shutil.copyfile(os.path.join(source_root_folder, group_folder+'_FDG', subject_folder), os.path.join(fusion_target_folder, 'source', 'PET', group_folder, subject_folder))

                            # if not os.path.exists(os.path.join(fusion_target_folder, 'source', 'MRI', group_folder)):
                            #     os.makedirs(os.path.join(fusion_target_folder, 'source', 'MRI', group_folder))
                            # shutil.copyfile(os.path.join(source_root_folder, group_folder+'_MRI_skull_strip', subject_folder), os.path.join(fusion_target_folder, 'source', 'MRI', group_folder, subject_folder))


                            # normalize(os.path.join(pvc_root_folder, group_folder, subject_folder))

                    


                




