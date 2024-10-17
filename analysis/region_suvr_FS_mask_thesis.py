

import os
import numpy as np
import nibabel as nib
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd
import json
import matplotlib.pyplot as plt

import glob
from PIL import Image

global root_folders, file_names, source_names
# Specify the root folders and the file names
root_folders = ["project/pvc/src/output/AD", "project/pvc/src/output/MCI", "project/pvc/src/output/CN"]
# file_names = ["PET.nii", "output_RBV_suvr.nii", "fusion.nii"]

file_names = ["PET.nii", "output_IY_suvr.nii", "fusion_best.nii", "Baseline.nii"]
source_names = ["PET", "PVC", "MRI-styled PET", "Baseline", "FS_mask", "FS_aseg_mni"]

# Function to load NIfTI file and return its data
def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

# Function to perform z-score normalization
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def p_value_notation(p_value):
    if p_value<0.001:
        return "***"
    elif p_value<0.01:
        return "**"
    elif p_value<0.05:
        return "*"
    else:
        return " "
        
def read_json(file_path):
    # with open(file_path, 'r', encoding='windows-1252') as file:
    #     lines = file.readlines()
    with open(file_path, 'r') as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

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

def box_plot(ax, datas, edge_color, fill_color, label, positions, y_min_limit):
    # print(np.array(datas).shape)
    bp = ax.boxplot(datas, patch_artist=True, positions=positions)
    ax.set_xticklabels([])
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    # for mean in bp['means']:
    #     mean.set_linestyle('--')

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    for i in range(len(positions)):
        # print(positions[i], source_names[i], label, np.array(datas[i]).shape)
        if fill_color=='skyblue':
            # print('skyblue')
            ax.text(positions[i], y_min_limit+0.05, source_names[i], ha='center', va='center', fontsize=28, color='black')
        ax.text(positions[i],  y_min_limit+0.15 , str(np.mean(datas[i]))[:4]+"\u00B1"+str(np.std(datas[i]))[:4], ha='center', va='center', fontsize=16, color=edge_color)
        ax.text(positions[i],  y_min_limit+0.25 , label, ha='center', va='center', fontsize=26, color=edge_color)
        # ax.axhline(y=np.mean(datas[i]), linestyle='--', color=edge_color, linewidth=2, xmin=positions[i]/11-0.05, xmax=positions[i]/11+0.05)
        # bp = ax.boxplot(data[i], patch_artist=True, positions=positions[i])
        # ax.set_xticklabels([])
        # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     plt.setp(bp[element], color=edge_color)

        # for patch in bp['boxes']:
        #     patch.set(facecolor=fill_color)
        
        
    return bp
def scatter_plot(ax, datas, marker_color, label, positions):
    # for pos, values in zip(positions, data):
        # ax.scatter([pos-0.5] * len(values), values, c=marker_color, label=label, alpha=0.7)
        # print(np.mean(values))
        
    for i in range(len(positions)):
        ax.scatter([positions[i]] * len(datas[i]), datas[i], c=marker_color, label=label, alpha=0.7)
        
def create_box_plot(data_all, group_name, edge_color, fill_color, x_title):
    # example_data1=data[0]
    # example_data2=data[1]
    # example_data3=data[2]
    

    fig, ax = plt.subplots(figsize=(14,16))

    # Set specific positions for each category
    # positions_bp1 = [0, 3, 6]
    # positions_bp2 = [1, 4, 7]
    # positions_bp3 = [2, 5, 8]

    # bp1 = box_plot(ax, example_data1, 'red', 'tan', 'AD', positions_bp1)
    # bp2 = box_plot(ax, example_data2, 'blue', 'cyan', 'MCI', positions_bp2)
    # bp3 = box_plot(ax, example_data3, 'darkgreen', 'lightgreen', 'CN', positions_bp3)
    
    # bp1 = box_plot(ax, data[0], edge_color[0], fill_color[0], group_name[0], [0, 3, 6])
    # bp2 = box_plot(ax, data[1], edge_color[1], fill_color[1], group_name[1], [1, 4, 7])
    # bp3 = box_plot(ax, data[2], edge_color[2], fill_color[2], group_name[2], [2, 5, 8])
    if x_title=='WM SUVr':
        y_min_limit=0.3
        y_max_limit=y_min_limit+1.5
    elif x_title=='GM SUVr':
        y_min_limit=0.6
        y_max_limit=y_min_limit+1.5
    elif x_title=='GM-WM ratio':
        y_min_limit=0.6
        y_max_limit=y_min_limit+2
    else:
        y_min_limit=0.5
        y_max_limit=y_min_limit+2
        
    for i in range(len(group_name)):
        # print(group_name[i])
        # print(data_all[i])
        # print(np.mean(data_all[i], axis=1))
        bp=box_plot(ax, data_all[i], edge_color[i], fill_color[i], group_name[i], [i, i+3, i+6], y_min_limit)
        scatter_plot(ax, data_all[i], edge_color[i], group_name[i], [i, i+3, i+6])
        
        

    # # Add scatter plots on top of the box plots
    # scatter_plot(ax, example_data1, 'red', 'AD', positions_bp1)
    # scatter_plot(ax, example_data2, 'blue', 'MCI', positions_bp2)
    # scatter_plot(ax, example_data3, 'darkgreen', 'CN', positions_bp3)
    
    ax.set_ylabel('SUVr',fontsize=30)
    ax.tick_params(axis='y', labelsize=20)
    # if x_title.find('cingulate')!=-1:
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_title(x_title, fontsize=35)
    # ax.set_xlabel(x_title, fontsize=35)
    if not os.path.exists("/home/linyunong/project/style_transfer/analysis/new_sessions/box_plot"):
        os.makedirs("/home/linyunong/project/style_transfer/analysis/new_sessions/box_plot")
    plt.savefig('project/style_transfer/analysis/new_sessions/box_plot/{}.png'.format(x_title))

def create_bland_altman_plot(data_all, region, cohens_d_data):
    group_names=['CN', 'MCI', 'AD']
    modality_names=["PET", "PVC", "MRI-styled PET", "Baseline"]
    color_index=[
        ['lightgreen', 'darkgreen', 'gray'],
        ['skyblue', 'navy', 'gray'],
        ['coral', 'maroon', 'gray']
    ]
    fontsize=20
    for i in range(len(group_names)):
       
        # print(group_names[i])    
            
        title_name=region+'('+group_names[i]+')'
        f, ax = plt.subplots(1, figsize = (12,12))
        mean1 = (data_all[i][1] + data_all[i][0])/2
        diff1 = data_all[i][1] - data_all[i][0]

        mean2 = (data_all[i][2] + data_all[i][0])/2
        diff2 = data_all[i][2] - data_all[i][0]
        
        mean3 = (data_all[i][3] + data_all[i][0])/2
        diff3 = data_all[i][3] - data_all[i][0]
        
        cohens_d_1=cohens_d(data_all[i][1], data_all[i][0])
        cohens_d_2=cohens_d(data_all[i][2], data_all[i][0])
        cohens_d_3=cohens_d(data_all[i][3], data_all[i][0])
        
        
        # row=[cohens_d_1, cohens_d_2]
        
        row=[region, group_names[i], cohens_d_1, cohens_d_2, cohens_d_3]
        cohens_d_data.append(row)
        print(row)
        # try:
        # cohens_d_data=cohens_d_data.append(row)
        # except AttributeError:
        #     cohens_d_df=row
        # try: 
        #     cohens_d_df = cohens_d_df.append(pd.Series(row, index=cohens_d_df.columns))#, ignore_index=True)
        # except UnboundLocalError:
        #     cohens_d_df=pd.DataFrame('Region':region, 'Group': i, 'PVC vs PET':cohens_d_1, 'MRI-styled PET vs PET':cohens_d_2)
        # use list instead dataframe
        

        # plot data
        # stats_info1="mean: {:.3f}, std: {:.3f}".format(np.mean(diff1), np.std(diff1))
        plt.scatter(mean1, diff1, s=100, alpha=0.7, color=color_index[i][0],label=modality_names[1])#+'(cohen\'s d={:.3f})'.format(cohens_d_1))
        # stats_info2="mean: {:.3f}, std: {:.3f}".format(np.mean(diff2), np.std(diff2))
        plt.scatter(mean2, diff2, s=100, alpha=0.7, color=color_index[i][1],label=modality_names[2])#+'(cohen\'s d={:.3f})'.format(cohens_d_2))
        plt.scatter(mean3, diff3, s=100, alpha=0.7, color=color_index[i][2],label=modality_names[3])#+'(cohen\'s d={:.3f})'.format(cohens_d_2))

        # plot mean difference line
        plt.axhline(np.mean(diff1), color=color_index[i][0], linestyle='-', linewidth = 3)
        plt.axhline(np.mean(diff2), color=color_index[i][1], linestyle='-', linewidth = 3)
        plt.axhline(np.mean(diff3), color=color_index[i][2], linestyle='-', linewidth = 3)
        
        # plt.text(1.75, np.mean(diff1)+0.01, 'mean diff:\n{:.3f}'.format(np.mean(diff1)), color=color_index[i][0], fontsize=fontsize+6, weight='bold')
        # # if region.find('SUVr')!=-1:
        # plt.text(1.75, np.mean(diff1)+ 1.96*np.std(diff1)+0.01, 'SD: {:.3f}'.format(np.std(diff1)), color=color_index[i][0], fontsize=fontsize+6, weight='bold')
        # else:
        #     plt.text(1.75, np.mean(diff1)+ 1.96*np.std(diff1)+0.01, 'SD: {:.3f}'.format(np.std(diff1)), color=color_index[i][0], fontsize=fontsize+6, weight='bold')
        
        
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
        
        # if region=='temporal':
            
        plt.xlabel('Mean of SUVR (output-PET)',fontsize = fontsize+14)
        plt.ylabel('Diff of SUVR (output-PET)',fontsize = fontsize+14)
    
        plt.title(group_names[i],fontsize = fontsize+30)
        
        plt.xlim(0.4,2.25)
        
        # if region=='WM SUVr':
        #     plt.ylim(-0.7,0.4)
        # elif region=='GM SUVr':
        #     plt.ylim(0,0.5)
        # elif region=="Temporal Lobe":
        #     plt.ylim(-0.3, 0.5)
        # # elif region=="Precuneus":
        # #     plt.ylim(-0.6, 0.5)  
        if region=="WM SUVr":
            plt.ylim(-0.7,0.4)    
        else:
            plt.ylim(-0.5, 0.5)
            
        # elif region.find('GM_WM_ratio')!=-1:
        #     plt.xlabel('Mean of GM-WM ratio (output-PET)',fontsize = fontsize+10)
        #     plt.ylabel('Diff of GM-WM ratio (output-PET)',fontsize = fontsize+10)
        #     plt.title(title_name,fontsize = fontsize+24)
        #     plt.xlim(0.8,1.6)
        #     plt.ylim(-0.4, 0.2)
            
        # else:
            
        #     plt.xlabel('Mean of SUVR in White matter(output-PET)',fontsize = fontsize+10)
        #     plt.ylabel('Diff of SUVR in White matter(output-PET)',fontsize = fontsize+10)
        
        #     plt.title(title_name,fontsize = fontsize+24)
        #     plt.xlim(0.6,1.2)
        #     plt.ylim(-0.2, 0.3)

        
        plt.xticks(fontsize=fontsize+6)
        plt.yticks(fontsize=fontsize+6)
        plt.savefig('/home/linyunong/project/style_transfer/analysis/new_sessions/bland_altman_thesis/'+title_name+'.png')
        plt.close()
    return cohens_d_data
    
if __name__ == "__main__":

    # # Create dictionaries to store data arrays for each file name
    data_arrays = {source_name: [] for source_name in source_names}
    #'Region', 'Group', 'PVC vs PET', 'MRI-styled PET vs PET'
    # cohens_d_df=[['Region', 'Group', 'PVC vs PET', 'MRI-styled PET vs PET']]
    cohens=[['Region', 'Group', 'PVC vs PET', 'MRI-styled PET vs PET', 'Baseline vs PET']]
    es_group=[['Region', 'Task', 'PET', 'PVC', 'MRI-styled PET']]
    
    
    # Iterate through root folders
    for root_folder in root_folders:
        # Iterate through subject folders
        for subject_folder in os.listdir(root_folder):
            subject_folder_path = os.path.join(root_folder, subject_folder)
            
            # =================================================
            # 1. Record PET, PVC, MRI-styled PET of each subject
            # Iterate through file names
            
            for f in range(len(file_names)):
                if file_names[f].find("016_S_4952")==-1:
                    file_name=file_names[f]
                    source_name=source_names[f]

                    # Construct the file path
                    file_path = os.path.join(subject_folder_path, file_name)
                    
                    # Load the NIfTI file and append its data to the corresponding array
                    data_arrays[source_name].append(load_nifti_file(file_path))
            # =================================================
            # 2. Record FreeSurfer mask of each subject
            file_path = os.path.join(root_folder.replace("pvc/src/output", "src/label_mni_new")+"_MRI", subject_folder, "aparc+aseg_mni.nii")

            # Load the NIfTI file and append its data to the corresponding array
            mask=load_nifti_file(file_path)
            data_arrays["FS_mask"].append(mask)
            
            
            file_path = os.path.join(root_folder.replace("pvc/src/output", "src/label_mni_new")+"_MRI", subject_folder, "aseg_mni.nii")

            # Load the NIfTI file and append its data to the corresponding array
            mask=load_nifti_file(file_path)
            data_arrays["FS_aseg_mni"].append(mask)
            
            # print(np.unique(mask))
            
            


    # Convert lists to NumPy arrays
    for source_name in source_names:
        data_arrays[source_name] = np.array(data_arrays[source_name])

    # Access the arrays as needed, for example:
    fusion_data = data_arrays["MRI-styled PET"]
    output_data = data_arrays["PVC"]
    pet_data = data_arrays["PET"]
    mask_data = data_arrays["FS_mask"]
    # print(fusion_data.shape)






    intra_group_dataframe_dict={"CN vs AD":pd.DataFrame(columns=['region',\
        'PET: p-value (avg. suvr)', 'PVC: p-value (avg. suvr)', 'MRI-styled PET: p-value (avg. suvr)', \
        'PET: p-value (avg. z-score)', 'PVC: p-value (avg. z-score)', 'MRI-styled PET: p-value (avg. z-score)', \
            
        'PET: CN (avg. suvr)','PVC: CN (avg. suvr)', 'MRI-styled PET: CN (avg. suvr)', \
        'PET: AD (avg. suvr)', 'PVC: AD (avg. suvr)', 'MRI-styled PET: AD (avg. suvr)', \
        'PET: CN (avg. z-score)', 'PVC: CN (avg. z-score)', 'MRI-styled PET: CN (avg. z-score)', \
        'PET: AD (avg. z-score)', 'PVC: AD (avg. z-score)', 'MRI-styled PET: AD (avg. z-score)'
        ]),\
            
        "CN vs MCI": pd.DataFrame(columns=['region', \
        'PET: p-value (avg. suvr)', 'PVC: p-value (avg. suvr)', 'MRI-styled PET: p-value (avg. suvr)', \
        'PET: p-value (avg. z-score)', 'PVC: p-value (avg. z-score)', 'MRI-styled PET: p-value (avg. z-score)', \
            
        'PET: CN (avg. suvr)', 'PVC: CN (avg. suvr)', 'MRI-styled PET: CN (avg. suvr)', \
        'PET: MCI (avg. suvr)', 'PVC: MCI (avg. suvr)', 'MRI-styled PET: MCI (avg. suvr)', \
        'PET: CN (avg. z-score)', 'PVC: CN (avg. z-score)', 'MRI-styled PET: CN (avg. z-score)', 
        'PET: MCI (avg. z-score)', 'PVC: MCI (avg. z-score)', 'MRI-styled PET: MCI (avg. z-score)']),\
            
        "MCI vs AD": pd.DataFrame(columns=['region',\
        'PET: p-value (avg. suvr)', 'PVC: p-value (avg. suvr)', 'MRI-styled PET: p-value (avg. suvr)',\
        'PET: p-value (avg. z-score)', 'PVC: p-value (avg. z-score)', 'MRI-styled PET: p-value (avg. z-score)',\
            
        'PET: MCI (avg. suvr)', 'PVC: MCI (avg. suvr)', 'MRI-styled PET: MCI (avg. suvr)', \
        'PET: AD (avg. suvr)', 'PVC: AD (avg. suvr)', 'MRI-styled PET: AD (avg. suvr)', \
        'PET: MCI (avg. z-score)', 'PVC: MCI (avg. z-score)', 'MRI-styled PET: MCI (avg. z-score)', \
        'PET: AD (avg. z-score)', 'PVC: AD (avg. z-score)', 'MRI-styled PET: AD (avg. z-score)'])}


    notation_dataframe_dict={
        "CN vs AD":pd.DataFrame(columns=['region',  'PET (avg. suvr)', 'PVC (avg. suvr)', 'MRI-styled PET (avg. suvr)', 'PET (avg. z-score)', 'PVC (avg. z-score)', 'MRI-styled PET (avg. z-score)']), \
        "CN vs MCI": pd.DataFrame(columns=['region',  'PET (avg. suvr)', 'PVC (avg. suvr)', 'MRI-styled PET (avg. suvr)', 'PET (avg. z-score)', 'PVC (avg. z-score)', 'MRI-styled PET (avg. z-score)']), \
        "MCI vs AD": pd.DataFrame(columns=['region',  'PET (avg. suvr)', 'PVC (avg. suvr)', 'MRI-styled PET (avg. suvr)', 'PET (avg. z-score)', 'PVC (avg. z-score)', 'MRI-styled PET (avg. z-score)'])}




    group_name=['CN', 'MCI', 'AD']
    edge_color=['darkgreen', 'navy', 'red']
    fill_color=['lightgreen', 'skyblue', 'tan']
    # ========================================
    # seperate region
    # region_json=read_json("/home/linyunong/project/src/label_mni_new/LUT_related.json")
    region_json={}
    # region_json["GM-WM ratio"]='GM-WM ratio'
    # region_json["GM SUVr"]='GM SUVr'
    # region_json["WM SUVr"]='WM SUVr'
    
    # ========================================
    
    region_json={
        # "Cingulate Gyrus": [1002, 2002, 1010, 2010, 1023, 2023, 1026, 2026],
        "Posterior Cingulate Cortex": [1023, 2023],
        "Temporal Lobe": [1009, 2009, 1015, 2015, 1030, 2030, 1033, 2033, 1034, 2034],
        "Parietal Lobe": [1008, 2008, 1029, 2029],
        # "Parahippocampus": [1016, 2016],
        "Frontal Lobe":[1003, 2003, 1012, 2012, 1014, 2014, 1027, 2027, 1028, 2028, 1032, 2032],
        # "Hippocampus":[17, 53],
        "Precuneus": [1025, 2025],
        
        "Posterior Cingulate Cortex": [1023, 2023],
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
        "Hippocampus":[17, 53]
        
        
    }
    

    region_json["GM SUVr"]='GM SUVr'
    region_json["WM SUVr"]='WM SUVr'
    # region_json["GM-WM ratio"]='GM-WM ratio'
    
    
    for region_num, region_dict in region_json.items():
        # =======================================
        # 1. FreeSurfer region
        if region_num!="GM-WM ratio" and region_num!="GM SUVr"and region_num!="WM SUVr":
            if region_num!="Hippocampus":
                # region=str(list(region_dict.keys())[0]).split("-")[-1]
                # lh_mask = np.isin(data_arrays["FS_mask"], region_dict[list(region_dict.keys())[0]])
                # rh_mask = np.isin(data_arrays["FS_mask"], region_dict[list(region_dict.keys())[1]])
                # lr_mask = lh_mask + rh_mask
                lr_mask = np.where(np.isin(data_arrays["FS_mask"].astype(int), region_dict), 1, 0)
            else:
                lr_mask = np.where(np.isin(data_arrays["FS_aseg_mni"].astype(int), region_dict), 1, 0)
                
            region=region_num    
            # print(lr_mask.shape, np.sum(lr_mask))
            data=[]
            for nii in data_arrays.keys():
                if nii.find("FS_mask")==-1 and nii.find("FS_aseg_mni")==-1:
                    # print(nii)
                    groups_suvr_array=data_arrays[nii]
                    # print(groups_suvr_array.shape)
                    # CN_suvr=np.sum(groups_suvr_array[100:] * lr_mask[100:], axis=(1, 2, 3))/np.sum(lr_mask[100:], axis=(1, 2, 3))
                    # MCI_suvr=np.sum(groups_suvr_array[50:100] * lr_mask[50:100], axis=(1, 2, 3))/np.sum(lr_mask[100:], axis=(1, 2, 3))
                    # AD_suvr=np.sum(groups_suvr_array[:50] * lr_mask[:50], axis=(1, 2, 3))/np.sum(lr_mask[100:], axis=(1, 2, 3))
                    
                    single_region_suvr = np.sum(groups_suvr_array * lr_mask, axis=(1, 2, 3))/np.sum(lr_mask, axis=(1, 2, 3)) # (150,1)
                        
                    #=============================
                    # SUVR
                    AD_suvr = single_region_suvr[:50]
                    MCI_suvr = single_region_suvr[50:100]
                    CN_suvr = single_region_suvr[100:]
                    
                    data.append([CN_suvr, MCI_suvr, AD_suvr])
                    # print(np.array(data).shape)
        # =======================================
        # 2. Gray-to-white matter ratio
        elif region_num=="GM-WM ratio":
            region='GM-WM ratio'
            data=[]
            for nii in data_arrays.keys():
                if nii.find("FS_mask")==-1 and nii.find("FS_aseg_mni")==-1:
                    # print(nii)
                    groups_suvr_array=data_arrays[nii]
                    
                    gm_mask = np.where(np.isin(data_arrays["FS_aseg_mni"].astype(int), [3, 42]), 1, 0)
                    wm_mask = np.where(np.isin(data_arrays["FS_aseg_mni"].astype(int), [2, 41]), 1, 0)
                    
                    gm_suvr = np.sum(groups_suvr_array * gm_mask, axis=(1, 2, 3))/np.sum(gm_mask, axis=(1, 2, 3)) # (150,1)
                    wm_suvr = np.sum(groups_suvr_array * wm_mask, axis=(1, 2, 3))/np.sum(wm_mask, axis=(1, 2, 3)) # (150,1)
                    # print(gm_suvr[:5])
                    # print(wm_suvr[:5])
                    
                    gm_wm_suvr=gm_suvr/wm_suvr
                    #=============================
                    # SUVR
                    AD_suvr = gm_wm_suvr[:50]
                    MCI_suvr = gm_wm_suvr[50:100]
                    CN_suvr = gm_wm_suvr[100:]
                    
                    data.append([CN_suvr, MCI_suvr, AD_suvr])
                    # print(np.array(data).shape)
        elif region_num=="GM SUVr":
            region='GM SUVr'
            data=[]
            for nii in data_arrays.keys():
                if nii.find("FS_mask")==-1 and nii.find("FS_aseg_mni")==-1:
                    # print(nii)
                    groups_suvr_array=data_arrays[nii]
                    
                    gm_mask = np.where(np.isin(data_arrays["FS_aseg_mni"].astype(int), [3, 42]), 1, 0)
                    
                    gm_suvr = np.sum(groups_suvr_array * gm_mask, axis=(1, 2, 3))/np.sum(gm_mask, axis=(1, 2, 3)) # (150,1)
                    
                    #=============================
                    # SUVR
                    AD_suvr = gm_suvr[:50]
                    MCI_suvr = gm_suvr[50:100]
                    CN_suvr = gm_suvr[100:]
                    
                    data.append([CN_suvr, MCI_suvr, AD_suvr])
                    # print(np.array(data).shape)
        elif region_num=="WM SUVr":
            region='WM SUVr'
            data=[]
            for nii in data_arrays.keys():
                if nii.find("FS_mask")==-1 and nii.find("FS_aseg_mni")==-1:
                    # print(nii)
                    groups_suvr_array=data_arrays[nii]
                    
                    wm_mask = np.where(np.isin(data_arrays["FS_aseg_mni"].astype(int), [2, 41]), 1, 0)
                    
                    wm_suvr = np.sum(groups_suvr_array * wm_mask, axis=(1, 2, 3))/np.sum(wm_mask, axis=(1, 2, 3)) # (150,1)
                    
                    #=============================
                    # SUVR
                    AD_suvr = wm_suvr[:50]
                    MCI_suvr = wm_suvr[50:100]
                    CN_suvr = wm_suvr[100:]
                    
                    data.append([CN_suvr, MCI_suvr, AD_suvr])
                    # print(np.array(data).shape)
                    
            
                   
            

        #0: PET, PVC, MRI-styled PET, 1:CN, MCI, AD
        # print(np.array(data).shape)
        data=np.swapaxes(np.array(data), 0, 1)
        # print(np.array(data).shape)
        data=data.tolist()
        
        # create_box_plot(data, group_name, edge_color, fill_color, str(aal_index)+'_'+aal_txt[str(aal_index)])
        # create_box_plot(data, group_name, edge_color, fill_color, region)
        cohens=create_bland_altman_plot(data, region, cohens)
        
        # data[i][j]; i->group; j->modality
        # group_names=['CN', 'MCI', 'AD']
        # modality_names=["PET", "PVC", "MRI-styled PET", "Baseline"]
        # TaskA: MCI vs CN; TaskB: AD vs CN; TaskC: AD vs MCI
        
        es_a_pet=cohens_d(data[0][0], data[1][0])
        es_a_pvc=cohens_d(data[0][1], data[1][1])
        es_a_fusion=cohens_d(data[0][2], data[1][2])
        
        es_b_pet=cohens_d(data[0][0], data[2][0])
        es_b_pvc=cohens_d(data[0][1], data[2][1])
        es_b_fusion=cohens_d(data[0][2], data[2][2])
        
        es_c_pet=cohens_d(data[1][0], data[2][0])
        es_c_pvc=cohens_d(data[1][1], data[2][1])
        es_c_fusion=cohens_d(data[1][2], data[2][2])
        
        es_group.append([region, 'MCI vs CN', es_a_pet, es_a_pvc, es_a_fusion])
        es_group.append([region, 'AD vs CN', es_b_pet, es_b_pvc, es_b_fusion])
        es_group.append([region, 'AD vs MCI', es_c_pet, es_c_pvc, es_c_fusion])
        print(es_group)
        # break
        
    df=pd.DataFrame(cohens[1:], columns=cohens[0])
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_gm_wm.csv")
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_ad_region.csv")
    df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_raw.csv")
    # df.to_xlsx("project/style_transfer/analysis/new_sessions/task1/cohens_d_raw.xlsx")
    
    df=pd.DataFrame(es_group[1:], columns=es_group[0])
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_gm_wm.csv")
    # df.to_csv("project/style_transfer/analysis/new_sessions/task1/cohens_d_ad_region.csv")
    df.to_csv("project/style_transfer/analysis/new_sessions/task1/effect_size_group_comparison.csv")
    # df.to_xlsx("project/style_transfer/analysis/new_sessions/task1/effect_size_group_comparison.xlsx")
    # df = pd.DataFrame(cohens[1:], columns=cohens[0])

    # # Specify the path where you want to save the Excel file
    # excel_file_path = "project/style_transfer/analysis/new_sessions/task1/cohens_d.xlsx"

    # # Write the DataFrame to an Excel file
    # df.to_excel(excel_file_path, index=False)

    # print(f"Excel file saved to {excel_file_path}")
    # color_index=[
    #     ['lightgreen', 'darkgreen'],
    #     ['skyblue', 'navy'],
    #     ['coral', 'maroon']
    # ]
    # data_cohen=np.array(cohens[1:])
    # print(data_cohen.shape)
    # num_groups = len(group_name)
    # num_regions = len(region_json.keys())

    # # Bar width
    # bar_width = 0.35

    # # X axis for the bar groups
    # x = np.arange(num_groups)

    # # Create subplots
    # fig, axs = plt.subplots(num_regions, figsize=(10, 8))

    # # Loop through regions
    # for i in range(num_regions):
    #     # Bar positions
    #     bar1 = x - bar_width / 2
    #     bar2 = x + bar_width / 2

    #     # Plot PVC vs PET
    #     axs[i].bar(bar1, data_cohen[i][0], bar_width, label='PVC vs PET', color=['lightgreen', 'skyblue', 'coral'])

    #     # Plot MRI-styled PET vs PET
    #     axs[i].bar(bar2, data_cohen[i][1], bar_width, label='MRI-styled PET vs PET', color=['darkgreen', 'navy', 'maroon'])

    #     # Set labels and title
    #     axs[i].set_xlabel('Group')
    #     axs[i].set_ylabel('Values')
    #     axs[i].set_title(list(region_json.keys())[i])
    #     axs[i].set_xticks(x)
    #     axs[i].set_xticklabels(group_name)
    #     axs[i].legend()

    # # Adjust layout
    # plt.tight_layout()

    # fig, ax = plt.subplots(figsize=(10, 8))
    # for i in range(num_regions):
    #     # Loop through groups
    #     for j in range(num_groups):
    #         # Bar positions for PVC vs PET
    #         bar1 = (i * num_groups + j) * 2
    #         # Bar positions for MRI-styled PET vs PET
    #         bar2 = bar1 + 1

    #         # Plot PVC vs PET
    #         ax.bar(bar1, pvc_values[j][i], bar_width, label=f'{regions[i]} - {groups[j]} (PVC vs PET)')

    #         # Plot MRI-styled PET vs PET
    #         ax.bar(bar2, pet_values[j][i], bar_width, label=f'{regions[i]} - {groups[j]} (MRI-styled PET vs PET)')

    # # Set labels and title
    # ax.set_xlabel('Region - Group')
    # ax.set_ylabel('Values')
    # ax.set_title('Interleaved Bar Chart')
    # ax.set_xticks(np.arange(num_regions * num_groups * 2))
    # ax.set_xticklabels([f'{regions[i]} - {groups[j]}' for i in range(num_regions) for j in range(num_groups)], rotation=45, ha='right')
    # ax.legend()

    # # Adjust layout
    # plt.tight_layout()

    # # Show plot
    # # plt.show()
    # # Show plot
    # plt.savefig("project/style_transfer/analysis/new_sessions/task1/cohens_d.png")

        
        # for nii in data_arrays.keys():
        #     if nii.find("FS_mask")==-1 and nii.find("FS_aseg_mni")==-1:
        #         groups_suvr_array=data_arrays[nii]
                
        #         suvr_mean_CN = np.mean(groups_suvr_array[100:], axis=(0), keepdims=True)
        #         suvr_std_CN = np.std(groups_suvr_array[100:], axis=(0), keepdims=True)
        #         z_map_group_CN = (groups_suvr_array[100:] - suvr_mean_CN) / suvr_std_CN
        #         z_map_group_MCI = (groups_suvr_array[50:100] - suvr_mean_CN) / suvr_std_CN
        #         z_map_group_AD = (groups_suvr_array[:50] - suvr_mean_CN) / suvr_std_CN
        #         mean_z_map_group_CN=np.mean(z_map_group_CN[:2], axis=0)
        #         mean_z_map_group_MCI=np.mean(z_map_group_MCI[:2], axis=0)
        #         mean_z_map_group_AD=np.mean(z_map_group_AD[:2], axis=0)

        #         # print(groups_suvr_array.shape, lr_mask.shape)
        #         # print(np.sum(groups_suvr_array * lr_mask, axis=(1, 2, 3)), np.sum(groups_suvr_array * lr_mask, axis=(1, 2, 3)).shape)
        #         # print(np.sum(lr_mask, axis=(1, 2, 3)), np.sum(lr_mask, axis=(1, 2, 3)).shape)
        #         single_region_suvr = np.sum(groups_suvr_array * lr_mask, axis=(1, 2, 3))/np.sum(lr_mask, axis=(1, 2, 3)) # (150,1)
                    
        #         #=============================
        #         # SUVR
        #         group_AD = single_region_suvr[:50]
        #         group_MCI = single_region_suvr[50:100]
        #         group_CN = single_region_suvr[100:]
        #         # print(group_AD)
        #         # print(group_AD.shape)

        #         # Calculate mean intensity for each group
        #         mean_suvr_group_AD = np.mean(group_AD)
        #         mean_suvr_group_MCI = np.mean(group_MCI)
        #         mean_suvr_group_CN = np.mean(group_CN)
        #         # print(mean_suvr_group_AD.shape)
                
        #         p_value_suvr_CN_AD=ttest_ind(group_AD, group_CN).pvalue
        #         p_value_suvr_CN_MCI=ttest_ind(group_MCI, group_CN).pvalue
        #         p_value_suvr_MCI_AD=ttest_ind(group_AD, group_MCI).pvalue
                
        #         #=============================
        #         # Z-score
                
                
        #         # group_AD = single_region_z_score[:50]
        #         # group_MCI = single_region_z_score[50:100]
        #         # group_CN = single_region_z_score[100:]
        #         # print(group_AD)
        #         # print(group_AD.shape)
                
        #         group_CN = np.sum(z_map_group_CN * lr_mask[100:], axis=(1, 2, 3))/np.sum(lr_mask[100:], axis=(1, 2, 3))
        #         group_MCI = np.sum(z_map_group_MCI * lr_mask[50:100], axis=(1, 2, 3))/np.sum(lr_mask[50:100], axis=(1, 2, 3))
        #         group_AD = np.sum(z_map_group_AD * lr_mask[:50], axis=(1, 2, 3))/np.sum(lr_mask[:50], axis=(1, 2, 3))
                

        #         # Calculate mean intensity for each group
        #         mean_z_score_group_AD = np.mean(group_AD)
        #         mean_z_score_group_MCI = np.mean(group_MCI)
        #         mean_z_score_group_CN = np.mean(group_CN)
        #         # print(mean_z_score_group_AD.shape)
                
                
                
        #         p_value_z_CN_AD=ttest_ind(group_AD, group_CN).pvalue
        #         p_value_z_CN_MCI=ttest_ind(group_MCI, group_CN).pvalue
        #         p_value_z_MCI_AD=ttest_ind(group_AD, group_MCI).pvalue
                
        #         if region in intra_group_dataframe_dict["CN vs AD"]['region'].values:

        #             intra_group_dataframe_dict["CN vs AD"].loc[intra_group_dataframe_dict["CN vs AD"]['region'] == region, \
        #             [nii+" (avg. suvr)", nii+" (avg. z-score)", nii+': CN (avg. suvr)', nii+': AD (avg. suvr)', nii+': CN (avg. z-score)', nii+': AD (avg. z-score)', nii+': p-value (avg. suvr)', nii+': p-value (avg. z-score)']] = \
        #             [p_value_notation(p_value_suvr_CN_AD), p_value_notation(p_value_z_CN_AD), mean_suvr_group_CN,  mean_suvr_group_AD, mean_z_score_group_CN, mean_z_score_group_AD, p_value_suvr_CN_AD, p_value_z_CN_AD]
                    
        #             intra_group_dataframe_dict["CN vs MCI"].loc[intra_group_dataframe_dict["CN vs MCI"]['region'] == region,\
        #             [nii+" (avg. suvr)", nii+" (avg. z-score)", nii+': CN (avg. suvr)', nii+': MCI (avg. suvr)', nii+': CN (avg. z-score)', nii+': MCI (avg. z-score)', nii+': p-value (avg. suvr)', nii+': p-value (avg. z-score)']] = \
        #             [p_value_notation(p_value_suvr_CN_MCI), p_value_notation(p_value_z_CN_MCI), mean_suvr_group_CN,  mean_suvr_group_MCI, mean_z_score_group_CN, mean_z_score_group_MCI, p_value_suvr_CN_MCI, p_value_z_CN_MCI]
                    
        #             intra_group_dataframe_dict["MCI vs AD"].loc[intra_group_dataframe_dict["MCI vs AD"]['region'] == region, \
        #             [nii+" (avg. suvr)", nii+" (avg. z-score)", nii+': MCI (avg. suvr)', nii+': AD (avg. suvr)', nii+': MCI (avg. z-score)', nii+': AD (avg. z-score)', nii+': p-value (avg. suvr)', nii+': p-value (avg. z-score)']] = \
        #             [p_value_notation(p_value_suvr_MCI_AD), p_value_notation(p_value_z_MCI_AD), mean_suvr_group_MCI,  mean_suvr_group_AD, mean_z_score_group_MCI, mean_z_score_group_AD, p_value_suvr_MCI_AD, p_value_z_MCI_AD]
                    
        #             notation_dataframe_dict["CN vs AD"].loc[notation_dataframe_dict["CN vs AD"]['region']== region, \
        #             ['region',  nii+" (avg. suvr)", nii+" (avg. z-score)"]] = \
        #             [region, p_value_notation(p_value_suvr_CN_AD), p_value_notation(p_value_z_CN_AD)]
        #             notation_dataframe_dict["CN vs MCI"].loc[notation_dataframe_dict["CN vs MCI"]['region']== region, \
        #             ['region',  nii+" (avg. suvr)", nii+" (avg. z-score)"]] = \
        #             [region, p_value_notation(p_value_suvr_CN_MCI), p_value_notation(p_value_z_CN_MCI)]
        #             notation_dataframe_dict["MCI vs AD"].loc[notation_dataframe_dict["MCI vs AD"]['region']== region, \
        #             ['region',  nii+" (avg. suvr)", nii+" (avg. z-score)"]] = \
        #             [region, p_value_notation(p_value_suvr_MCI_AD), p_value_notation(p_value_z_MCI_AD)]

        #         else:

        #             new_row = {'region': region, nii+': CN (avg. suvr)': mean_suvr_group_CN, nii+': AD (avg. suvr)': mean_suvr_group_AD, nii+': CN (avg. z-score)': mean_z_score_group_CN, nii+': AD (avg. z-score)': mean_z_score_group_AD, \
        #             nii+': p-value (avg. suvr)': p_value_suvr_CN_AD, nii+': p-value (avg. z-score)': p_value_z_CN_AD,\
        #             nii+" (avg. suvr)": p_value_notation(p_value_suvr_CN_AD), nii+" (avg. z-score)": p_value_notation(p_value_z_CN_AD)}
        #             intra_group_dataframe_dict["CN vs AD"]=intra_group_dataframe_dict["CN vs AD"].append(new_row, ignore_index=True)
                    
        #             new_row = {'region': region, nii+': CN (avg. suvr)': mean_suvr_group_CN, nii+': MCI (avg. suvr)': mean_suvr_group_MCI, nii+': CN (avg. z-score)': mean_z_score_group_CN, nii+': MCI (avg. z-score)': mean_z_score_group_MCI, \
        #             nii+': p-value (avg. suvr)': p_value_suvr_CN_MCI, nii+': p-value (avg. z-score)': p_value_z_CN_MCI,\
        #             nii+" (avg. suvr)": p_value_notation(p_value_suvr_CN_MCI), nii+" (avg. z-score)": p_value_notation(p_value_z_CN_MCI)}
        #             intra_group_dataframe_dict["CN vs MCI"]=intra_group_dataframe_dict["CN vs MCI"].append(new_row, ignore_index=True)
                    
        #             new_row = {'region': region, nii+': MCI (avg. suvr)': mean_suvr_group_MCI, nii+': AD (avg. suvr)': mean_suvr_group_AD, nii+': MCI (avg. z-score)': mean_z_score_group_MCI, nii+': AD (avg. z-score)': mean_z_score_group_AD,\
        #             nii+': p-value (avg. suvr)': p_value_suvr_MCI_AD, nii+': p-value (avg. z-score)': p_value_z_MCI_AD,\
        #             nii+" (avg. suvr)": p_value_notation(p_value_suvr_MCI_AD), nii+" (avg. z-score)": p_value_notation(p_value_z_MCI_AD)}
        #             intra_group_dataframe_dict["MCI vs AD"]=intra_group_dataframe_dict["MCI vs AD"].append(new_row, ignore_index=True)

        #             new_row = {'region': region, nii+" (avg. suvr)": p_value_notation(p_value_suvr_CN_AD), nii+" (avg. z-score)": p_value_notation(p_value_z_CN_AD)}
        #             notation_dataframe_dict["CN vs AD"]=notation_dataframe_dict["CN vs AD"].append(new_row, ignore_index=True)

        #             new_row = {'region': region, nii+" (avg. suvr)": p_value_notation(p_value_suvr_CN_MCI), nii+" (avg. z-score)": p_value_notation(p_value_z_CN_MCI)}
        #             notation_dataframe_dict["CN vs MCI"]=notation_dataframe_dict["CN vs MCI"].append(new_row, ignore_index=True)

        #             new_row = {'region': region, nii+" (avg. suvr)": p_value_notation(p_value_suvr_MCI_AD), nii+" (avg. z-score)": p_value_notation(p_value_z_MCI_AD)}
        #             notation_dataframe_dict["MCI vs AD"]=notation_dataframe_dict["MCI vs AD"].append(new_row, ignore_index=True)
                    
        #             print("CN vs AD")
        #             print(intra_group_dataframe_dict["CN vs AD"])
        #             print(notation_dataframe_dict["CN vs AD"])

        #             print("CN vs MCI")
        #             print(intra_group_dataframe_dict["CN vs MCI"])
        #             print(notation_dataframe_dict["CN vs MCI"])

        #             print("MCI vs AD")
        #             print(intra_group_dataframe_dict["MCI vs AD"])
        #             print(notation_dataframe_dict["MCI vs AD"])

        #         # intra_group_dataframe_dict["CN vs AD"].to_csv("/home/linyunong/project/pvc/stats/stats_CN_AD.csv", index=False)    
        #         if not os.path.exists("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance"):
        #             os.makedirs("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance")
        #         intra_group_dataframe_dict["CN vs AD"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/stats_CN_AD.xlsx", index=False)
        #         notation_dataframe_dict["CN vs AD"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/notation_CN_AD.xlsx", index=False)

        #         intra_group_dataframe_dict["CN vs MCI"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/stats_CN_MCI.xlsx", index=False)
        #         notation_dataframe_dict["CN vs MCI"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/notation_CN_MCI.xlsx", index=False)

        #         intra_group_dataframe_dict["MCI vs AD"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/stats_MCI_AD.xlsx", index=False)
        #         notation_dataframe_dict["MCI vs AD"].to_excel("/home/linyunong/project/style_transfer/analysis/new_sessions/regional_significance/notation_MCI_AD.xlsx", index=False)
    
    






