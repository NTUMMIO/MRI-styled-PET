# MRI-styled PET: A Dual Modality Fusion Approach to PET Partial Volume Correction

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

- ## Installation
1. Clone the repository:
```bash
git clone https://github.com/NTUMMIO/MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement.git
```

2. Install dependencies:
```bash
cd MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement
conda env create -f environment.yaml
 ```

## Usage
1. Make sure your image directory is structured as below, 
```
your_image_folder
├─── subject001
│    └─── PET.nii
│    └─── MRI.nii
│    └─── aparc.nii (Optional)
│    └─── aparc+aseg.nii (Optional)
├─── patient002
│    └─── PET.nii
│    └─── MRI.nii
├─── ...
```

Under each subject's folder, there must exist both PET.nii and MRI.nii to run without errors.
**(Recommended)** If possible, the FreeSurfer recon-all -all pipeline should be run in advance. Refer to [run_reconall_parallel.sh](main/run_reconall_parallel.sh) on how to run recon-all on all subjects under your image folder. If you have yet installed FreeSurfer on your OS, please refer to the [official FreeSurfer documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). 

However, if FreeSurfer was not available to you, we also support two alternatives.

### Option 1: SPM segmentation

blah, blah, blah, ...

### Option 2: Pseudo segmented tissue maps estimated directly from MR images (i.e., the MR intensity-derived approach)

blah, blah, blah, ...

2. Load the pretrained MRI-styled PET model
To inference with input shape (112, 112, 112), you can run
```bash
cd main
python inference.py --config ../src/experiments/best.json --segmentation 1 --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --input_directory /your_image_dir 
```

Or with input shape (224, 224, 224), you can run
```bash
cd main
python inference.py --config ../src/experiments/best.json --segmentation 1 --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best_224.pth --input_directory /your_image_dir 
```
3. Train your own MRI-styled PET model

(Beta)
blah, blah, blah, ...
