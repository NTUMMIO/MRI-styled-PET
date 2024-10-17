# calculate the slice wise psnr and ssim of two .nii (src and {model}) in /home/linyunong/project/style_transfer/suvr_nii

import os
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import csv
import argparse
import json
# from image_fusion_metrics.avg_gradient import metricsAvg_gradient
# from image_fusion_metrics.mutual_information import metricsMutinf
# from image_fusion_metrics.cnr import metricsCNR
import matplotlib.pyplot as plt


def calc_metrics_PET(fused_folder, source_folder, mask_folder):

    subjects_folder=os.listdir(fused_folder)
    subject_list=[]
    psnr_list=[]
    ssim_list=[]
    # avg_grad_list=[]
    mi_list=[]
    
    
    
    for subject_folder in subjects_folder:

        fused_nii=nib.load(os.path.join(fused_folder,subject_folder)).get_fdata()
        source_nii=nib.load(os.path.join(source_folder,subject_folder)).get_fdata()[:, :, :, 0]
        mask_nii=nib.load(os.path.join(mask_folder, subject_folder[:-4], "aparc+aseg_mni.nii")).get_fdata()
        if fused_nii.shape != source_nii.shape:
            raise ValueError("Images must have the same shape")

        num_slices = fused_nii.shape[-1]

        psnr_values = []
        ssim_values = []
        # avg_grad_values = []

        for slice_idx in range(num_slices):
            # Extract slices
            slice1 = fused_nii[..., slice_idx]
            slice2 = source_nii[..., slice_idx]
            data_range = np.max([slice1.max(), slice2.max()])

            # Compute PSNR
            psnr = peak_signal_noise_ratio(slice1, slice2, data_range=data_range)
            

            # Compute SSIM
            ssim = structural_similarity(slice1, slice2, data_range=data_range)
            if ~np.isnan(psnr) and ~np.isnan(ssim) and ~np.isinf(psnr) and ~np.isinf(ssim):
                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # print(slice_idx, psnr, ssim)
            

            # avg_grad=metricsAvg_gradient(slice1)
            # # print(avg_grad, metricsAvg_gradient(np.random(shape=slice1.shape)))
            # avg_grad_values.append(avg_grad)
            
        subject_list.append(subject_folder)
        psnr_list.append(sum(psnr_values) / len(psnr_values))
        ssim_list.append(sum(ssim_values) / len(ssim_values))
        # avg_grad_list.append(sum(avg_grad_values) / len(avg_grad_values))
        
        # mi=metricsMutinf(source_nii, fused_nii, mask_nii)
        # mi_list.append(mi)
        # print(subject_folder, mi)
    return subject_list, psnr_list, ssim_list#, mi_list   

def calc_metrics_MRI(fused_folder, source_folder, mask_folder):

    subjects_folder=os.listdir(fused_folder)
    # print(fused_folder, len(subjects_folder))
    # print(subjects_folder)
    # subjects_folder_MRI=os.listdir(source_folder)
    
    subject_list=[]
    psnr_list=[]
    ssim_list=[]
    # avg_grad_list=[]
    mi_list=[]
    
    for subject_folder in subjects_folder:
        # for subject_folder_MRI in subjects_folder_MRI:
        #     if subject_folder_MRI.find(subject_folder[:-4])!=-1:
                # print(subject_folder)
                fused_nii=nib.load(os.path.join(fused_folder,subject_folder)).get_fdata()
                source_nii=nib.load(os.path.join(source_folder,subject_folder)).get_fdata()#[:, :, :, 0]
                mask_nii=nib.load(os.path.join(mask_folder, subject_folder[:-4], "aparc+aseg_mni.nii")).get_fdata()
                # print(os.path.join(mask_folder, subject_folder[:-4], "aparc+aseg_mni.nii"))
                if fused_nii.shape != source_nii.shape:
                    # print(fused_nii.shape, source_nii.shape)
                    raise ValueError("Images must have the same shape")
                    

                num_slices = fused_nii.shape[-1]

                psnr_values = []
                ssim_values = []
                # avg_grad_values = []

                for slice_idx in range(num_slices):
                    # Extract slices
                    slice1 = fused_nii[..., slice_idx]
                    slice2 = source_nii[..., slice_idx]
                    data_range = np.max([slice1.max(), slice2.max()])

                    # Compute PSNR
                    psnr = peak_signal_noise_ratio(slice1, slice2, data_range=data_range)
                    # psnr_values.append(psnr)

                    # Compute SSIM
                    ssim = structural_similarity(slice1, slice2, data_range=data_range)
                    # ssim_values.append(ssim)

                    if ~np.isnan(psnr) and ~np.isnan(ssim) and ~np.isinf(psnr) and ~np.isinf(ssim):
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        # print(slice_idx, psnr, ssim)

                    # print(slice_idx, psnr, ssim)

                    
                    # avg_grad=metricsAvg_gradient(slice1)
                    # avg_grad_values.append(avg_grad)
                subject_list.append(subject_folder)
                psnr_list.append(sum(psnr_values) / len(psnr_values))
                ssim_list.append(sum(ssim_values) / len(ssim_values))
                # avg_grad_list.append(sum(avg_grad_values) / len(avg_grad_values))
        
        # mi=metricsMutinf(source_nii, fused_nii, mask_nii)
        # mi_list.append(mi)
    return subject_list, psnr_list, ssim_list#, mi_list  


def calc_metrics_PVC(fused_folder, source_folder, mask_folder):

    subjects_folder=os.listdir(fused_folder)
    subjects_folder_MRI=os.listdir(source_folder)
    
    subject_list=[]
    psnr_list=[]
    ssim_list=[]
    # avg_grad_list=[]
    mi_list=[]
    
    for subject_folder in subjects_folder:
        # print(subject_folder)
        # subject_folder='003_S_4441.nii'

        fused_nii=nib.load(os.path.join(fused_folder,subject_folder)).get_fdata()
        source_nii=nib.load(os.path.join(source_folder,subject_folder)).get_fdata()#[:, :, :, 0]

        # source_nii=nib.load(os.path.join(source_folder,subject_folder[:-4],'output_IY_8mm_suvr.nii')).get_fdata()
        mask_nii=nib.load(os.path.join(mask_folder, subject_folder[:-4], "aparc+aseg_mni.nii")).get_fdata()
        # mask_tissue=nib.load(os.path.join(mask_folder, subject_folder[:-4], "label_4d_full.nii")).get_fdata()
        # print(np.unique(mask_tissue.astype(int)), mask_tissue.shape)
        
        # nii_file=open(os.path.join(mask_folder, subject_folder[:-4], 'output_GTM.nii'), "r")
        # gtm_list=[]
        # for nii in nii_file:
        #     print(nii.split('\t'))
        #     if nii.split('\t')[0].find('REGION')==-1:
        #         gtm_list.append(float(nii.split('\t')[1]))

        # print(np.sum(mask_tissue[:, :, :, 2])* gtm_list[2], np.sum(mask_tissue[:, :, :, 2]))
        # print(np.sum(mask_tissue[:, :, :, 3])* gtm_list[3], np.sum(mask_tissue[:, :, :, 3]))
        # print(np.max(gtm_list))
        # gtm_list=gtm_list/np.max(gtm_list)
        
                
        # gtm_np=np.array(gtm_list).reshape(1, 1, 1, -1)
        # ideal_tracer=mask_tissue* gtm_np
        # ideal_tracer=np.sum(ideal_tracer, axis=-1)
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(ideal_tracer[:,:,40])
        # plt.subplot(1,3,2)
        # plt.imshow(ideal_tracer[:,40,:])
        # plt.subplot(1,3,3)
        # plt.imshow(ideal_tracer[40,:,:])
        # plt.colorbar()
        # plt.savefig('project/style_transfer/analysis/image_fusion_metrics/test.png')

        
        if fused_nii.shape != source_nii.shape:
            print(fused_nii.shape, source_nii.shape)
            raise ValueError("Images must have the same shape")
            

        num_slices = fused_nii.shape[-1]

        psnr_values = []
        ssim_values = []
        avg_grad_values = []
        crn_list = []

        for slice_idx in range(num_slices):
            # Extract slices
            slice1 = fused_nii[..., slice_idx]
            slice2 = source_nii[..., slice_idx]
            data_range = np.max([slice1.max(), slice2.max()])

            # Compute PSNR
            psnr = peak_signal_noise_ratio(slice1, slice2, data_range=data_range)
            # psnr_values.append(psnr)

            # Compute SSIM
            ssim = structural_similarity(slice1, slice2, data_range=data_range)
            # ssim_values.append(ssim)

            # print(slice_idx, psnr, ssim)
            
            if ~np.isnan(psnr) and ~np.isnan(ssim) and ~np.isinf(psnr) and ~np.isinf(ssim):
                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # print(slice_idx, psnr, ssim)

            # avg_grad=metricsAvg_gradient(slice1)
            # avg_grad_values.append(avg_grad)
        subject_list.append(subject_folder)
        psnr_list.append(sum(psnr_values) / len(psnr_values))
        ssim_list.append(sum(ssim_values) / len(ssim_values))
        # avg_grad_list.append(sum(avg_grad_values) / len(avg_grad_values))
        
        # mi=metricsMutinf(source_nii, fused_nii, mask_nii)
        # mi_list.append(mi)
        
        # cnr=metricsCNR(source_nii, fused_nii, mask_nii)
        # crn_list.append(cnr)
    return subject_list, psnr_list, ssim_list#, mi_list  


def calc_metrics_MRI_and＿PET_against_PVＣ(fused_folder, pvc_folder, source_folder):


    subjects_folder=os.listdir(fused_folder)
    print(pvc_folder)
    subject_list=[]
    psnr_pet_list=[]
    ssim_pet_list=[]
    psnr_mri_list=[]
    ssim_mri_list=[]
    # avg_grad_list=[]
    mi_list=[]
    
    

    for subject_folder in subjects_folder:
        print(os.path.join(pvc_folder, subject_folder[:-4],'output_IY_8mm_suvr.nii'))
        pvc_nii=nib.load(os.path.join(pvc_folder, subject_folder[:-4],'output_IY_8mm_suvr.nii')).get_fdata()
        source_nii_pet=nib.load(os.path.join(source_folder, subject_folder)).get_fdata()
        source_nii_mri=nib.load(os.path.join(source_folder.replace('PET', 'MRI'), subject_folder)).get_fdata()


        if pvc_nii.shape != source_nii_pet.shape:
            raise ValueError("Images must have the same shape")
        print(pvc_nii.shape, source_nii_pet.shape, source_nii_mri.shape)
        num_slices = pvc_nii.shape[-1]

        psnr_pet_values = []
        ssim_pet_values = []
        psnr_mri_values = []
        ssim_mri_values = []
        # avg_grad_values = []

        for slice_idx in range(num_slices):
            # Extract slices
            slice1 = pvc_nii[..., slice_idx]
            slice_pet = source_nii_pet[..., slice_idx]
            slice_mri = source_nii_mri[..., slice_idx]

            data_range = np.max([slice1.max(), slice_pet.max()])
            data_range = np.max([slice1.max(), slice_mri.max()])


            # Compute PSNR
            psnr_pet = peak_signal_noise_ratio(slice1, slice_pet, data_range=data_range)
            psnr_pet_values.append(psnr_pet)
            psnr_mri = peak_signal_noise_ratio(slice1, slice_mri, data_range=data_range)
            psnr_mri_values.append(psnr_mri)

            # Compute SSIM
            ssim_pet = structural_similarity(slice1, slice_pet, data_range=data_range)
            ssim_pet_values.append(ssim_pet)
            ssim_mri = structural_similarity(slice1, slice_mri, data_range=data_range)
            ssim_mri_values.append(ssim_mri)

            # avg_grad=metricsAvg_gradient(slice1)
            # # print(avg_grad, metricsAvg_gradient(np.random(shape=slice1.shape)))
            # avg_grad_values.append(avg_grad)
            
        subject_list.append(subject_folder)
        psnr_pet_list.append(sum(psnr_pet_values) / len(psnr_pet_values))
        ssim_pet_list.append(sum(ssim_pet_values) / len(ssim_pet_values))
        psnr_mri_list.append(sum(psnr_mri_values) / len(psnr_mri_values))
        ssim_mri_list.append(sum(ssim_mri_values) / len(ssim_mri_values))
        # avg_grad_list.append(sum(avg_grad_values) / len(avg_grad_values))
        
        # mi=metricsMutinf(source_nii, fused_nii, mask_nii)
        # mi_list.append(mi)
        # print(subject_folder, mi)
    return subject_list, psnr_pet_list, ssim_pet_list, psnr_mri_list, ssim_mri_list

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model_folders', nargs='+', help='select models to calculate psnr and ssim', required=True)
    # args = parser.parse_args()
    
    
    # model_folders=['sc_att_prior_boundary_cerebellum', 'sc_att_prior_boundary_cerebellum_conversion']#'sc_att_full', 'vanilla_att_prior', 'sc_att_prior', 'self_att_prior', 'cross_att_prior', 'swinunet_vanilla_att_prior', 'swinunet_sc_att_prior', 'swinunet_self_att_prior', 'swinunet_cross_att_prior']
    pet_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/PET"
    mri_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/MRI"

    root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion"
    mask_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/roi"
    pvc_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY_8mm"
    # model_folders=os.listdir(root_folder)
    model_folders=['Baseline']#'Baseline',	'ConfigA',	'ConfigB',	'ConfigC_1',	'ConfigC1_cross',	'ConfigC1_vanilla',	
    #['ConfigB_1']#, 'ConfigB1_cross2', 'ConfigB1_cs', 'ConfigB1_cross', 'ConfigB1_vanilla']#, , 'ConfigC1_swin']#'ConfigB1_cross','ConfigB1_vanilla', 'ConfigB1_swin', 'ConfigB1_cross_swin','ConfigB1_vanilla_swin']#['Baseline', 'ConfigA', 'ConfigB', 'ConfigB_1', 'ConfigB_3', 'ConfigC', 'ConfigC_1']
    for model_folder in model_folders:
        fused_folder = os.path.join(root_folder, model_folder)
        if model_folder.find('src')==-1:
            fold_folders = os.listdir(fused_folder)
            for fold_folder in fold_folders:
                # fused folder, i.e. "/home/linyunong/project/style_transfer/output_images/src/fold0/fused"
                # fused_folder = os.path.join(fused_folder, fold_folder)
                model_num_folders = os.listdir(os.path.join(fused_folder, fold_folder))
                for model_num_folder in model_num_folders:
                    # if model_folder=='sc_att_prior': #or model_num_folder=='2024-1-6-105337':
                    
                        # fused_folder = os.path.join(fused_folder, model_num_folder)
                        group_folders=os.listdir(os.path.join(fused_folder, fold_folder, model_num_folder))
                        
                        # print(group_folders)
                        for group_folder in group_folders:
                            
                            # print(os.path.join(fused_folder, fold_folder, model_num_folder, group_folder))
                            # print(os.path.join(root_folder, fold_folder, "fused", group_folder))
                            output_folder=os.path.join(fused_folder, fold_folder, model_num_folder, group_folder)
                            pet_folder=os.path.join(pet_root_folder, group_folder)#src_root_folder, 'src', fold_folder, "PET", group_folder)
                            mri_folder=os.path.join(mri_root_folder, group_folder)
                            mask_folder=os.path.join(mask_root_folder, group_folder)#.split("_")[0])
                            pvc_foler=os.path.join(pvc_root_folder, group_folder)
             

                            # subject_list, psnr_pet_list, ssim_pet_list, psnr_mri_list, ssim_mri_list = calc_metrics_MRI_and＿PET_against_PVＣ(output_folder, '/home/linyunong/project/pvc/src/output/{}'.format(group_folder[:-8]), source_folder)

                            print(output_folder)
                            subject_list, psnr_list_PET, ssim_list_PET  = calc_metrics_PET(output_folder, pet_folder, mask_folder)
                            print("finished PET ")
                            subject_list, psnr_list_MRI, ssim_list_MRI  = calc_metrics_MRI(output_folder, mri_folder, mask_folder)
                            print("finished MRI")
                            subject_list, psnr_list_PVC, ssim_list_PVC  = calc_metrics_PVC(output_folder, pvc_foler, mask_folder)
                            print("finished PVC")

                            # for i in range(len(subject_list)):
                            #     print(subject_list[i], psnr_pet_list[i], ssim_pet_list, psnr_mri_list, ssim_mri_list)
                            
                            csv_filename="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/raw_quantitative_metrics_old_pipeline/"+model_folder+"/"+fold_folder+"/"+ model_num_folder+ "/"+ group_folder+".csv"
                            if not os.path.exists("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/raw_quantitative_metrics_old_pipeline/"+model_folder+"/"+fold_folder+"/"+ model_num_folder):
                                os.makedirs("/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/raw_quantitative_metrics_old_pipeline/"+model_folder+"/"+fold_folder+"/"+ model_num_folder)
                            print(csv_filename)
                            
                            with open(csv_filename, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(['Subject', 'SSIM_PET', 'SSIM_PVC', 'SSIM_MRI', 'PSNR_PET', 'PSNR_PVC', 'PSNR_MRI'])
                                for i in range(len(subject_list)):
                                    row=[subject_list[i], ssim_list_PET[i], ssim_list_PVC[i], ssim_list_MRI[i], psnr_list_PET[i], psnr_list_PVC[i], psnr_list_MRI[i]]#, psnr_list_PVC[i], ssim_list_PVC[i],  mi_PVC[i]]
                                    writer.writerow(row)

                            # csv_filename="/home/linyunong/project/style_transfer/analysis/new_sessions/raw_metrics/Reference/"+ group_folder+".csv"
                            # if not os.path.exists("/home/linyunong/project/style_transfer/analysis/new_sessions/raw_metrics/Reference"):
                            #     os.makedirs("/home/linyunong/project/style_transfer/analysis/new_sessions/raw_metrics/Reference")
                            # print(csv_filename)
                            
                            # with open(csv_filename, 'w', newline='') as csvfile:
                            #     writer = csv.writer(csvfile)
                            #     for i in range(len(subject_list)):
                            #         row=[subject_list[i], psnr_pet_list[i], ssim_pet_list[i], psnr_mri_list[i], ssim_mri_list[i]]#, psnr_list_PVC[i], ssim_list_PVC[i],  mi_PVC[i]]
                            #         writer.writerow(row)
                        
                        