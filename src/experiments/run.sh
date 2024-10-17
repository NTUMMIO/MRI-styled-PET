# --------------------------------------------------
# Exp1-1: Native space validation
# --------------------------------------------------
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --segmentation 1 --suvr

# --------------------------------------------------
# Exp1-2: SUV validation
# --------------------------------------------------
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --segmentation 1 



# --------------------------------------------------
# --------------------------------------------------
# Exp2: Resolution experiment
# --------------------------------------------------
# 224*224
# --------------------------------------------------

# Use FreeSurfer-simulated PET as PET input and pseudo segmentation (derived through MR intensity) as anatomical input
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best_224.pth --segmentation 1 --suvr --input_directory ../src/native_space_mr_crop/ --simulation 1

# Use SPM-simulated PET as PET input and pseudo segmentation (derived through MR intensity) as anatomical input
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best_224.pth --segmentation 1 --suvr --input_directory ../src/native_space_mr_crop/ --simulation 2

# --------------------------------------------------
# 112*112
# --------------------------------------------------

# Use FreeSurfer-simulated PET as PET input and pseudo segmentation (derived through MR intensity) as anatomical input
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --segmentation 1 --suvr --input_directory ../src/native_space_mr_crop_downsample/ --simulation 1

# Use SPM-simulated PET as PET input and pseudo segmentation (derived through MR intensity) as anatomical input
python inference.py --config ../src/experiments/best.json --resume_fusion_checkpoint ../src/checkpoint/model_fusion_best.pth --segmentation 1 --suvr --input_directory ../src/native_space_mr_crop_downsample/ --simulation 2

