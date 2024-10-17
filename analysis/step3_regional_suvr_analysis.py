import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_regional_suvr(df, df_all, i, image, roi1, roi2):
    # region_dict={
    #     "Posterior Cingulate Cortex": [1023, 2023],
    #     "Temporal Lobe": [1009, 2009, 1015, 2015, 1030, 2030, 1033, 2033, 1034, 2034],
    #     "Parietal Lobe": [1008, 2008, 1029, 2029],
    #     "Frontal Lobe":[1003, 2003, 1012, 2012, 1014, 2014, 1027, 2027, 1028, 2028, 1032, 2032],
    #     "Gray Matter": [3, 42],
    #     "White Matter": [2, 41],
    # }
    
    region_dict={

        "Posterior Cingulate Cortex": [1023, 2023],
        "Temporal Lobe": [1009, 2009, 1015, 2015, 1030, 2030, 1033, 2033, 1034, 2034],
        "Parietal Lobe": [1008, 2008, 1029, 2029],
        "Frontal Lobe":[1003, 2003, 1012, 2012, 1014, 2014, 1027, 2027, 1028, 2028, 1032, 2032],
        "Caudal Anterior Cingulate Cortex": [1002, 2002],
        "Rostral Anterior Cingulate Cortex": [1026, 2026],
        "Isthmus of Cingulate Gyrus": [1010, 2010],
        "Inferior Temporal": [1009, 2009],
        "Middle Temporal": [1015, 2015],
        "Superior Temporal": [1030, 2030],
        "Temporal Pole": [1033, 2033],
        "Transverse Temporal": [1034, 2034],
        "Inferior Parietal": [1008, 2008],
        "Superior Parietal": [1029, 2029],
        "Caudal Middle Frontal": [1003, 2003],
        "Lateral Orbito Frontal": [1012, 2012],
        "Medial Orbito Frontal": [1014, 2014],
        "Rostral Middle Frontal": [1027, 2027],
        "Superior Frontal": [1028, 2028],
        "Frontal Pole": [1032, 2032],
        "Parahippocampus": [1016, 2016],
        "Hippocampus":[17, 53],
        
        "Gray Matter": [3, 42],
        "White Matter": [2, 41],
        
    }
    
    num=len(df_all)-1
    print(num)
    for region, codes in region_dict.items():
        if region.find("Matter")==-1:
            mask = np.where(np.isin(roi1.astype(int), region_dict[region]), 1, 0)
        else:
            mask = np.where(np.isin(roi2.astype(int), region_dict[region]), 1, 0)
            
        df.loc[i, region]=np.sum(image*mask)/np.sum(mask)
        df_all.loc[num, region]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi1.astype(int), region_dict["Posterior Cingulate Cortex"]), 1, 0)
    # df.loc[i, "Posterior Cingulate Cortex"]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi1.astype(int), region_dict["Temporal Lobe"]), 1, 0)
    # df.loc[i, "Temporal Lobe"]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi1.astype(int), region_dict["Parietal Lobe"]), 1, 0)
    # df.loc[i, "Parietal Lobe"]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi1.astype(int), region_dict["Frontal Lobe"]), 1, 0)
    # df.loc[i, "Frontal Lobe"]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi2.astype(int), region_dict["Gray Matter"]), 1, 0)
    # df.loc[i, "Gray Matter"]=np.sum(image*mask)/np.sum(mask)

    # mask = np.where(np.isin(roi2.astype(int), region_dict["White Matter"]), 1, 0)
    # df.loc[i, "White Matter"]=np.sum(image*mask)/np.sum(mask)

    return df, df_all

def cohens_d(group1, group2):
    """
    Calculates Cohen's d effect size between two groups.

    Args:
        group1 (list or np.array): Data from the first group.
        group2 (list or np.array): Data from the second group.

    Returns:
        float: Cohen's d effect size.
    """
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Calculate pooled standard deviation
    n1 = len(group1)
    n2 = len(group2)
    pooled_var = ((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    # Calculate Cohen's d
    cohen_d = (mean2 - mean1) / pooled_std

    return cohen_d

def calculate_group_wise_suvr_difference(data):

    modality_names=["PET", "PVC", "Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]
    cohensd_group_difference = [["Region", "Task", "PET", "PVC", "Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]] #"MRI-styled PET"

    for region in ROIs:   

        cn_mci_row=[region, "MCI vs CN"]
        cn_ad_row=[region, "AD vs CN"]
        mci_ad_row=[region, "AD vs MCI"]
        for i in range(len(modality_names)):
            
            
            cn=np.array(data["CN"][modality_names[i]][region].to_list())
            mci=np.array(data["MCI"][modality_names[i]][region].to_list())        
            ad=np.array(data["AD"][modality_names[i]][region].to_list())
            
            task_cn_mci=cohens_d(cn, mci)
            task_cn_ad=cohens_d(cn, ad)
            task_mci_ad=cohens_d(mci, ad)

            cn_mci_row.append(task_cn_mci)
            cn_ad_row.append(task_cn_ad)
            mci_ad_row.append(task_mci_ad)
            
        cohensd_group_difference.append(cn_mci_row)
        cohensd_group_difference.append(cn_ad_row)
        cohensd_group_difference.append(mci_ad_row)
    
    df=pd.DataFrame(np.array(cohensd_group_difference)[1:], columns=np.array(cohensd_group_difference)[0])
    df.to_csv(os.path.join(cohensd_root_folder, "cohensd_group_wise_difference.csv"))


    
    
    

        
        
        
    
    #         es_a_pet=cohens_d(data[0][0], data[1][0])
    #         es_a_pvc=cohens_d(data[0][1], data[1][1])
    #         es_a_fusion=cohens_d(data[0][2], data[1][2])
            
    #         es_b_pet=cohens_d(data[0][0], data[2][0])
    #         es_b_pvc=cohens_d(data[0][1], data[2][1])
    #         es_b_fusion=cohens_d(data[0][2], data[2][2])
            
    #         es_c_pet=cohens_d(data[1][0], data[2][0])
    #         es_c_pvc=cohens_d(data[1][1], data[2][1])
    #         es_c_fusion=cohens_d(data[1][2], data[2][2])
            
    #         es_group.append([region, 'MCI vs CN', es_a_pet, es_a_pvc, es_a_fusion])
    #         es_group.append([region, 'AD vs CN', es_b_pet, es_b_pvc, es_b_fusion])
    #         es_group.append([region, 'AD vs MCI', es_c_pet, es_c_pvc, es_c_fusion])
    #         print(es_group)
    #     # break
        
    # df=pd.DataFrame(cohens[1:], columns=cohens[0])
    # # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_gm_wm.csv")
    # # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_ad_region.csv")
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_raw.csv")
    # # df.to_xlsx("project/style_transfer/analysis/new_sessions/task1/cohens_d_raw.xlsx")
    
    # df=pd.DataFrame(es_group[1:], columns=es_group[0])
    # # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_gm_wm.csv")
    # # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_ad_region.csv")
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/effect_size_group_comparison.csv")
    
def calculate_corrective_effects(data):

    
    # config_folders=["Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]
    ROIs=["Posterior Cingulate Cortex",
        "Temporal Lobe",
        "Parietal Lobe",
        "Frontal Lobe",
        "Gray Matter",
        "White Matter"]
    group_names=["CN", "MCI", "AD"]

    for i in range(len(config_experiment_folders)):
        config_experiment_folder=config_experiment_folders[i].replace('.csv', '')
        cohensd_corrective_effect=[["Region", "Group", "PVC vs PET", "MRI-styled PET vs PET", "Baseline vs PET"]]
        for region in ROIs:
            
            modality_names=["PET", "PVC", config_experiment_folder, "Baseline_2024-9-10-113411"]
            for i in range(len(group_names)):
                
                pet=np.array(data[group_names[i]][modality_names[0]][region].to_list())
                pvc=np.array(data[group_names[i]][modality_names[1]][region].to_list())        
                mr_styled_pet=np.array(data[group_names[i]][modality_names[2]][region].to_list())
                baseline=np.array(data[group_names[i]][modality_names[3]][region].to_list())
                
                
                # mean1 = (pvc + pet)/2
                # diff1 = pvc - pet

                # mean2 = (mr_styled_pet + pet)/2
                # diff2 = mr_styled_pet - pet
                
                # mean3 = (baseline + pet)/2
                # diff3 = baseline - pet
                
                cohens_d_1=cohens_d(pet, pvc)
                cohens_d_2=cohens_d(pet, mr_styled_pet)
                cohens_d_3=cohens_d(pet, baseline)
                row=[region, group_names[i], cohens_d_1, cohens_d_2, cohens_d_3]
                cohensd_corrective_effect.append(row)

        df=pd.DataFrame(np.array(cohensd_corrective_effect)[1:], columns=np.array(cohensd_corrective_effect)[0])
        df.to_csv(os.path.join(cohensd_root_folder, "cohensd_corrective_effects/{}.csv".format(config_experiment_folder)))
    
def create_bland_altman_plot(data):
    # config_folders=["Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]
    ROIs=["Posterior Cingulate Cortex",
        "Temporal Lobe",
        "Parietal Lobe",
        "Frontal Lobe",
        "Gray Matter",
        "White Matter"]

    for i in range(len(config_experiment_folders)):
        config_folder=config_experiment_folders[i].replace('.csv', '')
        if config_folder.find('Baseline')!=-1 or config_folder.find('_pseudo')!=-1 or config_folder.find('_real')!=-1:
            for region in ROIs:

                group_names=["CN", "MCI", "AD"]
                modality_names=["PET", "PVC", config_folder, "Baseline_2024-9-10-113411"]

                title_names=["PET", "PVC-PET", "Baseline", "MRI-styled PET"]

                color_index=[
                    ['lightgreen', 'darkgreen', 'gray'],
                    ['skyblue', 'navy', 'gray'],
                    ['coral', 'maroon', 'gray']
                ]
                fontsize=24
                
                for i in range(len(group_names)):

                        
                    title_name=region+'('+group_names[i]+')'
                    f, ax = plt.subplots(1, figsize = (17,15))
                    print(config_folder, title_name)
                    pet=np.array(data[group_names[i]][modality_names[0]][region].to_list())
                    pvc=np.array(data[group_names[i]][modality_names[1]][region].to_list())        
                    mr_styled_pet=np.array(data[group_names[i]][modality_names[3]][region].to_list())
                    baseline=np.array(data[group_names[i]][modality_names[2]][region].to_list())
                    
                    
                    
                    mean1 = (pvc + pet)/2
                    diff1 = pvc - pet

                    mean2 = (baseline + pet)/2
                    diff2 = baseline - pet
                    
                    mean3 = (mr_styled_pet + pet)/2
                    diff3 = mr_styled_pet - pet
                    
                    # cohens_d_1=cohens_d(pvc, pet)
                    # cohens_d_2=cohens_d(mr_styled_pet, pet)
                    # cohens_d_3=cohens_d(baseline, pet)
                    # row=[region, group_names[i], cohens_d_1, cohens_d_2, cohens_d_3]
                    # cohens_d_data.append(row)
                    
                    # plot data
                    # stats_info1="mean: {:.3f}, std: {:.3f}".format(np.mean(diff1), np.std(diff1))
                    plt.scatter(mean1, diff1, s=100, alpha=0.7, color=color_index[i][0],label=title_names[1])#+'(cohen\'s d={:.3f})'.format(cohens_d_1))
                    # stats_info2="mean: {:.3f}, std: {:.3f}".format(np.mean(diff2), np.std(diff2))
                    plt.scatter(mean2, diff2, s=100, alpha=0.7, color=color_index[i][1],label=title_names[2])#+'(cohen\'s d={:.3f})'.format(cohens_d_2))
                    plt.scatter(mean3, diff3, s=100, alpha=0.7, color=color_index[i][2],label=title_names[3])#+'(cohen\'s d={:.3f})'.format(cohens_d_2))

                    # plot mean difference line
                    plt.axhline(np.mean(diff1), color=color_index[i][0], linestyle='-', linewidth = 3)
                    plt.axhline(np.mean(diff2), color=color_index[i][1], linestyle='-', linewidth = 3)
                    plt.axhline(np.mean(diff3), color=color_index[i][2], linestyle='-', linewidth = 3)
                
                    plt.text(1.85, np.mean(diff1)+0.01, 'mean:{:.3f}\n SD: {:.3f}'.format(np.mean(diff1), np.std(diff1)), color=color_index[i][0], fontsize=fontsize+6, weight='bold')
                    plt.text(1.85, np.mean(diff2)+0.01, 'mean:{:.3f}\n SD: {:.3f}'.format(np.mean(diff2), np.std(diff2)), color=color_index[i][1], fontsize=fontsize+6, weight='bold')
                    plt.text(1.85, np.mean(diff3)+0.01, 'mean:{:.3f}\n SD: {:.3f}'.format(np.mean(diff3), np.std(diff3)), color=color_index[i][2], fontsize=fontsize+6, weight='bold')
                    
                    
                    
                    # plt.text(0.45, np.mean(diff2)+0.1, 'mean diff: {:.3f}'.format(np.mean(diff2)), color=color_index[i][1], fontsize=fontsize+6, weight='bold')
                    # if region.find('WM SUVr')==-1 and region.find('ratio')==-1:
                    #     plt.text(0.45, np.mean(diff2)+ 1.96*np.std(diff2)+0.01, 'SD: {:.3f}'.format(np.std(diff2)), color=color_index[i][1], fontsize=fontsize+6, weight='bold')
                    # else:
                    #     plt.text(0.45, np.mean(diff2)+ 0.15, 'SD: {:.3f}'.format(np.std(diff2)), color=color_index[i][1], fontsize=fontsize+6, weight='bold')
                    
                    
                    # plot limits of agreement
                    plt.axhline(np.mean(diff1) + 1.96*np.std(diff1), color=color_index[i][0], linestyle='--', linewidth = 3)
                    plt.axhline(np.mean(diff1) - 1.96*np.std(diff1), color=color_index[i][0], linestyle='--', linewidth = 3)
                    
                    plt.axhline(np.mean(diff2) + 1.96*np.std(diff2), color=color_index[i][1], linestyle='--', linewidth = 3)
                    plt.axhline(np.mean(diff2) - 1.96*np.std(diff2), color=color_index[i][1], linestyle='--', linewidth = 3)
                    
                    plt.axhline(np.mean(diff3) + 1.96*np.std(diff3), color=color_index[i][2], linestyle='--', linewidth = 3)
                    plt.axhline(np.mean(diff3) - 1.96*np.std(diff3), color=color_index[i][2], linestyle='--', linewidth = 3)
                    
                    ax.legend(fontsize = fontsize+12)
                    
                        
                    plt.xlabel('Mean of SUVR (output-PET)',fontsize = fontsize+14)
                    plt.ylabel('Diff of SUVR (output-PET)',fontsize = fontsize+14)
                
                    plt.title(group_names[i],fontsize = fontsize+30)
                    
                    plt.xlim(0.4,2.25)
                    
                    # if region=="WM SUVr":
                    #     plt.ylim(-0.7,0.4)    
                    # else:
                    #     plt.ylim(-0.5, 0.5)
                        
                    plt.ylim(-1, 1)
                    
                        

                    
                    plt.xticks(fontsize=fontsize+6)
                    plt.yticks(fontsize=fontsize+6)
                    if not os.path.exists('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/bland_altman/{}'.format(config_folder)):
                        os.makedirs('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/bland_altman/{}'.format(config_folder))
                        
                    plt.savefig('/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/bland_altman/{}/{}.png'.format(config_folder, title_name))
                    plt.close()
    # return cohens_d_data
    
def calculate_regional_suvr(data):
# ===============================
# Calculate regional SUVr of PET and PVC
# ===============================
    group_folders=os.listdir(os.path.join(pet_root_folder))
    group_folders=['AD', 'MCI', 'CN']
    
    df_pet_all=pd.DataFrame(columns=['ID', 'Group'])
    df_pvc_all=pd.DataFrame(columns=['ID', 'Group'])
    
    for group_folder in group_folders:
    
        subject_folders=os.listdir(os.path.join(pet_root_folder, group_folder))

        if subject_folders.count("016_S_4952.nii")> 0:
            print("remove")
            subject_folders.remove("016_S_4952.nii")

        df_pet=pd.DataFrame(columns=['Subject', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])
        df_pvc=pd.DataFrame(columns=['Subject', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])
        for i in range(len(subject_folders)):
        
            subject_folder=subject_folders[i]
            
            

            pet_current_subject_path=os.path.join(pet_root_folder, group_folder, subject_folder)
            # pet_current_subject_path=os.path.join(pet_root_folder, group_folder, subject_folder, 'PET.nii')

            print(pet_current_subject_path)
            pvc_current_subject_path=os.path.join(pvc_root_folder, group_folder, subject_folder)
            # pvc_current_subject_path=os.path.join(pvc_root_folder, group_folder, subject_folder, 'output_IY_suvr.nii')

            roi1_current_subject_path=os.path.join(roi_root_folder, group_folder, subject_folder.split('.')[0], "aparc+aseg_mni.nii")
            roi2_current_subject_path=os.path.join(roi_root_folder, group_folder, subject_folder.split('.')[0], "aseg_mni.nii")

            pet=nib.load(pet_current_subject_path).get_fdata()#[:, :, :, 0]
            pvc=nib.load(pvc_current_subject_path).get_fdata()
            roi1=nib.load(roi1_current_subject_path).get_fdata()
            roi2=nib.load(roi2_current_subject_path).get_fdata()

            df_pet.loc[i, 'Subject']=subject_folder.split('.')[0]
            df_pvc.loc[i, 'Subject']=subject_folder.split('.')[0]
            
            df_pet_all.loc[i, 'ID']=subject_folder.split('.')[0]
            df_pvc_all.loc[i, 'ID']=subject_folder.split('.')[0]
            
            df_pet_all.loc[i, 'Group']=group_folder
            df_pvc_all.loc[i, 'Group']=group_folder
            

            df_pet, df_pet_all=calc_regional_suvr(df_pet, df_pet_all, i, pet, roi1, roi2)
            df_pvc, df_pvc_all=calc_regional_suvr(df_pvc, df_pvc_all, i, pvc, roi1, roi2)
            
            
            

        data[group_folder]["PET"]=df_pet
        data[group_folder]["PVC"]=df_pvc
        # print(df_pet)
        # print(df_pvc)
        df_pet.to_csv(os.path.join(regional_suvr_root_folder, group_folder, "PET.csv"))
        df_pvc.to_csv(os.path.join(regional_suvr_root_folder, group_folder, "PVC.csv"))
        print(df_pvc_all)
    df_pet_all.to_excel(os.path.join(regional_suvr_root_folder, "PET.xlsx"))
    df_pvc_all.to_excel(os.path.join(regional_suvr_root_folder, "PVC.xlsx"))


# ===============================
# Calculate regional SUVr of each config (in fold0)
# ===============================

    # config_folders=os.listdir(fusion_root_folder)
    config_folders=["ConfigA", "ConfigB"]
    
    for config_folder in config_folders:
    
        df_all=pd.DataFrame(columns=['ID', 'Group'])
        
        fold_folders=os.listdir(os.path.join(fusion_root_folder, config_folder))
        fold_folders=['fold0']
        for fold_folder in fold_folders:
            experiment_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder))
            for experiment_folder in experiment_folders:
                # if experiment_folder.find('9-12')==-1 and experiment_folder.find('9-13')==-1:
                # if experiment_folder.find('train')!=-1:
                
                    group_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder))
                    for group_folder in group_folders:
                        subject_folders=os.listdir(os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder, group_folder))
                        if subject_folders.count("016_S_4952.nii")> 0:
                            print("remove")
                            subject_folders.remove("016_S_4952.nii")

                        df=pd.DataFrame(columns=['Subject', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])

                        for i in range(len(subject_folders)):
                            
                            subject_folder=subject_folders[i]
                            
                            fusion_current_subject_path=os.path.join(fusion_root_folder, config_folder, fold_folder, experiment_folder, group_folder, subject_folder)
                            print(fusion_current_subject_path)
                            roi1_current_subject_path=os.path.join(roi_root_folder, group_folder, subject_folder.split('.')[0], "aparc+aseg_mni.nii")
                            roi2_current_subject_path=os.path.join(roi_root_folder, group_folder, subject_folder.split('.')[0], "aseg_mni.nii")


                            fusion=nib.load(fusion_current_subject_path).get_fdata()
                            
                            roi1=nib.load(roi1_current_subject_path).get_fdata()
                            roi2=nib.load(roi2_current_subject_path).get_fdata()

                            df.loc[i, "Subject"]=subject_folder.split('.')[0]
                            
                            df_all.loc[i, 'ID']=subject_folder.split('.')[0]
                            df_all.loc[i, 'Group']=group_folder
                            # print(subject_folder.split('.')[0], fusion.shape, roi1.shape, roi2.shape)
                            df, df_all=calc_regional_suvr(df, df_all, i, fusion, roi1, roi2)
                            
                        
                        data[group_folder][config_folder+"_"+experiment_folder]=df
                        df.to_csv(os.path.join(regional_suvr_root_folder, group_folder, "{}_{}.csv".format(config_folder, experiment_folder)))
                        config_experiment_folders.append(group_folder+"_"+experiment_folder)
                    df_all.to_excel(os.path.join(regional_suvr_root_folder, "{}_{}.xlsx".format(config_folder, experiment_folder)))
    return data

def load_regional_suvr(data):
    group_folders=os.listdir(regional_suvr_root_folder)
    for group_folder in group_folders:
        # config_experiment_folders=["Baseline_2024-9-10-113411.csv","ConfigB_ConfigB_real.csv"]
        # config_experiment_folders=os.listdir(os.path.join(regional_suvr_root_folder, group_folder))
        config_experiment_folders=["PET.csv", "PVC.csv", "Baseline_2024-9-10-113411.csv", "ConfigC_ConfigC_real.csv", "ConfigD_ConfigD_real.csv"]
        for config_experiment_filename in config_experiment_folders:
            df=pd.read_csv(os.path.join(regional_suvr_root_folder, group_folder, config_experiment_filename), index_col=0)
            config_experiment_name=config_experiment_filename.replace('.csv', '')
            # print(df)
            data[group_folder][config_experiment_name]=df
    return data, config_experiment_folders

def calculate_regional_suvr_differences(data):
    group_folders=list(data.keys())
    # config_folders=["PVC", "Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]
    # config_folders=list(data['CN'].keys())
    # config_folders.remove('PET')
    for group_folder in group_folders:
        df_mean=pd.DataFrame(columns=['Modality', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])
        df_std=pd.DataFrame(columns=['Modality', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])

        df_mean_abs=pd.DataFrame(columns=['Modality', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])
        df_std_abs=pd.DataFrame(columns=['Modality', 'Posterior Cingulate Cortex', 'Temporal Lobe', 'Parietal Lobe', 'Frontal Lobe', 'Gray Matter', 'White Matter'])
        # print(df_mean)

        for i in range(len(config_experiment_folders)):
            config_experiment_folder=config_experiment_folders[i].replace('.csv', '')
            df_config=data[group_folder][config_experiment_folder]
            df_pet=data[group_folder]["PET"]
            # print(df_config)
            # print(df_pet)
            df_absolute_difference=df_pet.copy()
            df_relative_difference=df_pet.copy()

            regions=list(df_config.columns)
            regions.remove('Subject')

            mean_statistics=[config_experiment_folder]
            std_statistics=[config_experiment_folder]

            mean_statistics_abs=[config_experiment_folder]
            std_statistics_abs=[config_experiment_folder]

            # print(regions)
            for region in regions:
                # print(df_config[region])
                # print(df_pet[region])
                df_absolute_difference[region] = df_config[region] - df_pet[region]
                df_relative_difference[region] = df_absolute_difference[region] / df_pet[region]
                # print(df_relative_difference[region].mean())
                # print(df_relative_difference[region].std())
                mean_statistics.append(df_relative_difference[region].mean())
                std_statistics.append(df_relative_difference[region].std())

                mean_statistics_abs.append(df_absolute_difference[region].mean())
                std_statistics_abs.append(df_absolute_difference[region].std())


            df_mean.loc[i]=mean_statistics
            df_std.loc[i]=std_statistics

            df_mean_abs.loc[i]=mean_statistics_abs
            df_std_abs.loc[i]=std_statistics_abs


            # print(df_absolute_difference)
            if not os.path.exists(os.path.join(regional_suvr_difference_folder, group_folder)):
                os.makedirs(os.path.join(regional_suvr_difference_folder, group_folder))
                os.makedirs(os.path.join(regional_suvr_difference_folder.replace('absolute', 'relative'), group_folder))

                # print(os.path.join(regional_suvr_difference_folder, group_folder, config_experiment_folder+'.csv'))
            df_absolute_difference.to_csv(os.path.join(regional_suvr_difference_folder, group_folder, config_experiment_folder+'.csv'))
            df_relative_difference.to_csv(os.path.join(regional_suvr_difference_folder.replace('absolute', 'relative'), group_folder, config_experiment_folder+'.csv'))
        df_mean.to_csv(os.path.join(regional_suvr_difference_folder.replace('absolute', 'relative'), '{}_mean.csv'.format(group_folder)), index_label="Modality")
        df_std.to_csv(os.path.join(regional_suvr_difference_folder.replace('absolute', 'relative'), '{}_std.csv'.format(group_folder)), index_label="Modality")

        df_mean_abs.to_csv(os.path.join(regional_suvr_difference_folder, '{}_mean.csv'.format(group_folder)), index_label="Modality")
        df_std_abs.to_csv(os.path.join(regional_suvr_difference_folder, '{}_std.csv'.format(group_folder)), index_label="Modality")

        

if __name__=='__main__':    

    # =============================================
    # Root folders
    # =============================================

    global fusion_root_folder, pet_root_folder, pvc_root_folder, roi_root_folder, regional_suvr_root_folder, regional_suvr_difference_folder, cohensd_root_folder, config_folders, config_experiment_folders

    fusion_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/fusion/experiment"

    # pet_root_folder="/home/linyunong/project/pvc/src/output"

    pet_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/source/PET"

    pvc_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/pvc/IY"

    # pvc_root_folder="/home/linyunong/project/pvc/src/output"

    roi_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/output_nifti/roi"
# ---------------------------------
# for classification
    regional_suvr_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/regional_suvr_classification"
# ---------------------------------
# for relative and absolute error and bland altman
    # regional_suvr_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/regional_suvr"
# ---------------------------------
    
    regional_suvr_difference_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/regional_absolute_suvr_difference"

    cohensd_root_folder="/home/linyunong/project/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement/analysis/results/cohensd"

    config_experiment_folders=[]

    # =============================================
    # Constants
    # =============================================
    # config_folders=["Baseline", "ConfigA", "ConfigB_ratio1", "ConfigC_ratio1", "ConfigD_ratio1"]
    config_folders=os.listdir(fusion_root_folder)
    # cohensd_corrective_effect=[["Region", "Group", "PVC vs PET", "MRI-styled PET vs PET", "Baseline vs PET"]]
    ROIs=["Posterior Cingulate Cortex",
        "Temporal Lobe",
        "Parietal Lobe",
        "Frontal Lobe",
        "Gray Matter",
        "White Matter"]
    
    # =============================================
    # SUVr data
    # =============================================
    data={
        "AD": {
            "PET":pd.DataFrame(),
            "PVC":pd.DataFrame(),
        },
        "MCI": {
            "PET":pd.DataFrame(),
            "PVC":pd.DataFrame(),
        },
        "CN": {
            "PET":pd.DataFrame(),
            "PVC":pd.DataFrame(),
        },

    }

    
    # ===============================
    # If regional SUVr was already calculated, then load from csv
    # ===============================
    # data, config_experiment_folders=load_regional_suvr(data)
    # print(data)
    # =============================
    # Otherwise, calculate again
    # ===============================
    data=calculate_regional_suvr(data)
    # print(config_experiment_folders)
    # ===============================
    # Step1: Calculate the absolute and relative differences in SUVr
    # ===============================
    # calculate_regional_suvr_differences(data)


    # ===============================
    # Step2: Bland Altman plot @ analysis/results/bland_altman)
    # -- Configuration
    #   -- Region(Group)
    # ===============================           
    # create_bland_altman_plot(data)

    # ===============================
    # Step3: Corrective effect (differences between PET and different correction methods) @ analysis/results/cohensd/cohensd_corrective_effects
    # -- Configuration
    #   -- Region 
    #       -- Group
    # =============================== 
    # calculate_corrective_effects(data)

        
    # ===============================
    # Step4: Group-wise SUVr difference (between any two groups of different modalities in various region) @ analysis/results/cohensd/cohensd_group_wise_difference.csv
    #   -- Region 
    #       -- Group
    # ===============================    
    # calculate_group_wise_suvr_difference(data)







                        
                
                        






