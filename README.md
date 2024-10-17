# MRI-styled-PET-Dual-modality-Fusion-for-PET-Image-Enhancement



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
To inference the MRI-styled PET model with input shape 112*112*112, run
```bash
cd main
python inference.py --config ../src/experiments/best.json --segmentation 1 --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --input_directory /your_image_dir 
```

Or with input shape 224*224*224, run
```bash
cd main
python inference.py --config ../src/experiments/best.json --segmentation 1 --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best_224.pth --input_directory /your_image_dir 
```

