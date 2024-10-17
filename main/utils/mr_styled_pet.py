# import argparse
from argparse import ArgumentParser
import json
import os
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import time
import torchvision.transforms as trns
from torch.optim import Adam, AdamW
from torch.autograd import Variable
import ray
from ray import tune



from utils.data_set import ADNIDataset
from utils.model.backbone.backbone import UNetPP, SwinUNet
from utils.utils import adjust_learning_rate
from utils.loss import BoundaryGradient, AdaptiveMSSSIM

class MR_styled_PET:
    def __init__(
        self,
        args: ArgumentParser = None,
        
    ):
        
        self.args=args
        
        #=================================================
        # Set args
        #=================================================
        with open(args.config, 'r') as file:
            config = json.load(file)
        for key, value in config.items():
            setattr(self.args, key, value)
        print("Loaded configuration from path: ", args.config)  
        
        #=================================================
        # Check CUDA
        #=================================================
        if self.args.cuda:
            print("Use gpu id: '{}'".format(self.args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus
            if not torch.cuda.is_available():
                    raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        self.args.seed = random.randint(1, 10000)
        # random.randint(1, 10000)
        print("Random Seed: ", self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        cudnn.benchmark = True 

        #=================================================
        # Assign output folder with current timestamps
        #=================================================
        self.output_folder="../model_checkpoint/"+self.args.folder+"/fold"+str(self.args.fold)+"/"+str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        print('Output folder: ', self.output_folder)

        
        
    def get_args(self):
        return self.args
    
    def get_output_folder(self):
        return self.output_folder
    
    def prepare_data(self):
        
        MRI_train_filenames=[]    
        PET_train_filenames=[]
        MRI_valid_filenames=[]    
        PET_valid_filenames=[]
        
        #===============================================
        # Load MRI training data  
        #===============================================
        f = open('../../src/fold_750/fold{}/train_MRI.txt'.format(str(self.args.fold)), 'r')  
        for line in f.readlines():
            MRI_train_filenames.append(str(line)[:-2])
            
        #===============================================
        # Load PET training data   
        #===============================================
        f = open('../../src/fold_750/fold{}/train_PET.txt'.format(str(self.args.fold)), 'r')
        for line in f.readlines():
            PET_train_filenames.append(str(line)[:-2])

        #===============================================
        # Load MRI validation data  
        #===============================================
        f = open('../../src/fold_750/fold{}/valid_MRI.txt'.format(str(self.args.fold)), 'r')  
        for line in f.readlines():
            MRI_valid_filenames.append(str(line)[:-2])
            
        #===============================================
        # Load PET validation data   
        #===============================================
        f = open('../../src/fold_750/fold{}/valid_PET.txt'.format(str(self.args.fold)), 'r')
        for line in f.readlines():
            PET_valid_filenames.append(str(line)[:-2])
            
        #===============================================
        # ADNIDataset and DataLoader
        #===============================================
        train_dataset = ADNIDataset(root0=PET_train_filenames, root1=MRI_train_filenames, transform=trns.Compose([trns.ToTensor()]))
        valid_dataset = ADNIDataset(root0=PET_valid_filenames, root1=MRI_valid_filenames, transform=trns.Compose([trns.ToTensor()]))
        
        self.trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=self.args.batchSize, shuffle=True)
        
        self.validloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=self.args.batchSize, shuffle=True)
    
    def build_model(self):
        self._load_backbone()
        if self.args.resume_fusion_checkpoint:
            self._load_weights()
        

    def start_training(self):
        def _trainer(config):
            for epoch in range(self.args.start_epoch, self.args.nEpochs):
            
                lr = adjust_learning_rate(epoch, self.args.lr, self.args.step)
                #===============================================
                # Set optimizer
                #===============================================
                if self.args.fusion_op == False:
                    self.optimizer = Adam([{'params': self.model_fusion.parameters()}, {'params': self.model_conversion.parameters()}], lr=self.args.lr)   
                else:
                    self.optimizer = Adam(self.model_fusion.parameters(), lr)
            
                #===============================================
                # Set model to train mode
                #===============================================
                self.model_fusion.train()
                if self.args.fusion_op == False:
                    self.model_conversion.train()
                print(config['ssim_smaller_win_pet'])
                self._set_loss_function(self.config)

                for iteration, data in enumerate(self.trainloader, 1):
                    #===============================================
                    # Data preprocessing
                    #===============================================
                    anatomical_input, PET_input, MRI_full = self._data_preprocessing(data, self.args.anatomical_input_option, self.args.cuda)

                    #===============================================
                    # Move model to CUDA
                    #===============================================
                    if self.args.cuda:
                        self.model_fusion.cuda()
                        if self.args.fusion_op == False:
                            self.model_conversion.cuda()

                    #===============================================
                    # Foward pass
                    #===============================================
                    fusion_output = self.model_fusion(PET_input, anatomical_input)
                    if self.args.fusion_op == False:
                        PET_pred, MRI_pred = self.model_conversion(fusion_output)

                    #===============================================
                    # Calculate individual losses
                    #===============================================
                    mse_loss_PET = self.mse_loss(fusion_output, PET_input)
                    mse_loss_MRI = self.mse_loss(fusion_output, anatomical_input)
                    adamsssim_loss_PET = self.adamsssim_loss_pet(fusion_output, PET_input)
                    adamsssim_loss_MRI = self.adamsssim_loss_mri(fusion_output, anatomical_input)
                    boundary_loss_MRI, boundary_loss_PET =self.boundary_loss(fusion_output, PET_input, anatomical_input)

                    if self.args.fusion_op == False:
                        mse_loss_PET += self.mse_loss(PET_pred, PET_input)
                        mse_loss_MRI += self.mse_loss(MRI_pred, MRI_full)
                        adamsssim_loss_PET += self.adamsssim_loss_pet(PET_pred, PET_input)
                        adamsssim_loss_MRI += self.adamsssim_loss_mri(MRI_pred, anatomical_input)
                        boundary_loss_MRI_conversion, boundary_loss_PET_conversion =self.boundary_loss(fusion_output, PET_input, anatomical_input)
                        boundary_loss_MRI += boundary_loss_MRI_conversion
                        boundary_loss_PET += boundary_loss_PET_conversion

                    #===============================================
                    # Calculate total loss
                    #===============================================
                    total_loss = \
                        mse_loss_PET*self.args.pixel_pet2mri + \
                        mse_loss_MRI + \
                        adamsssim_loss_PET*self.args.ssim_pet2mri + \
                        adamsssim_loss_MRI + \
                        boundary_loss_PET*self.args.boundary_pet2mri + \
                        boundary_loss_MRI
                    
                    #===============================================    
                    # Compute gradient
                    #===============================================    
                    self.optimizer.zero_grad()

                    #===============================================    
                    # Backward pass
                    #===============================================    
                    total_loss.backward()
                    
                    #===============================================    
                    # Report loss to RayTune
                    #===============================================
                    ray.train.report({'total_loss': total_loss.item(), 
                        'pixel PET': mse_loss_PET.item(), 
                        'pixel MRI': mse_loss_MRI.item(), 
                        'ssim PET': adamsssim_loss_PET.item(), 
                        'ssim MRI': adamsssim_loss_MRI.item(), 
                        'boundary PET': boundary_loss_MRI.item(), 
                        'boundary MRI': boundary_loss_PET.item()})
                    
                    #===============================================    
                    # Update weights
                    #===============================================
                    self.optimizer.step()

                #===============================================
                # Save one image per epoch
                #===============================================
                # save_images_per_epoch_conversion(MRI_full.cpu().detach().numpy(), outputs.cpu().detach().numpy(), MRI_input.cpu().detach().numpy(), MRI_pred.cpu().detach().numpy(), PET_input.cpu().detach().numpy(), PET_pred.cpu().detach().numpy(), output_folder, epoch, True, iteration, data[2])
            return
        #===============================================
        # Start RayTune
        #===============================================
        ray.init(num_cpus=1, num_gpus=1, local_mode=True)

        config = {
            "ssim_smaller_win_pet": tune.loguniform(0.1, 0.5),
            "ssim_larger_win_pet": tune.loguniform(0.1, 0.4),
            
            "ssim_smaller_win_mri": tune.loguniform(0.1, 0.5),
            "ssim_larger_win_mri": tune.loguniform(0.1, 0.4),
        }

        analysis = tune.run(
            _trainer,
            config=config,
            resources_per_trial={"cpu": 1, "gpu": 1},
            metric="total_loss",
            mode="min",  # Use "min" mode for minimizing the loss
        )

        # Print the best hyperparameters
        best_config = analysis.get_best_config(metric="total_loss", mode="min")
        print("Best hyperparameters:", best_config)

        # Optionally, you can print more information about the analysis, such as the best trial
        print("Best trial:")
        print(analysis.get_best_trial(metric="total_loss", mode="min"))

        # Shut down Ray
        ray.shutdown()

    # def run_validation(self):
    

        


    def _data_preprocessing(self, data, anatomical_input_option=3, cuda=True):

        batch_pet=data[0]
        batch_mri=data[1]

        #===============================================
        # PET input
        #===============================================
        PET_input = Variable(batch_pet[:,0,:,:],requires_grad=True)
        PET_input = PET_input[:,None,:,:]
        
        #===============================================
        # Full T1-MRI
        #===============================================
        MRI_full = Variable(batch_mri[:,0,:,:], requires_grad=True)
        MRI_full = MRI_full[:,None,:,:]
        
        #===============================================
        # Anatomical input
        # 0 = full MRI
        # 1 = PET coarse
        # 2 = PET fine   
        # 3 = = predefined 4:1:0
        # 4 = predefined 5:1:0
        # 5 = GTM coarse
        # 6 = GTM fine
        #===============================================
        anatomical_input = Variable(batch_mri[:,anatomical_input_option,:,:], requires_grad=True)
        anatomical_input = anatomical_input[:,None,:,:]

        #===============================================
        # Weighted mask
        #===============================================
        weighted_mask = Variable(batch_mri[:,3,:,:], requires_grad=True)
        weighted_mask = weighted_mask[:,None,:,:]

        #===============================================
        # device to CUDA
        #===============================================
        if cuda:
            PET_input = PET_input.cuda()
            anatomical_input = anatomical_input.cuda()
            MRI_full = MRI_full.cuda()

        return anatomical_input, PET_input, MRI_full, weighted_mask
                           
    def _set_loss_function(self, config):
        
        #===============================================
        # Set loss function
        #===============================================
        self.mse_loss = torch.nn.MSELoss()
        self.boundary_loss=BoundaryGradient()
        # print(config['ssim_larger_win_mri'].dtype), config['ssim_larger_win_pet'].dtype)
        self.MRI_sets_winSize_weighting=[(7, config['ssim_larger_win_mri']), (5, 1-config['ssim_larger_win_mri']-config['ssim_smaller_win_mri']), (3, config['ssim_smaller_win_mri'])]
        self.PET_sets_winSize_weighting=[(7, config['ssim_larger_win_pet']), (5, 1-config['ssim_larger_win_pet']-config['ssim_smaller_win_pet']), (3, config['ssim_smaller_win_pet'])]
        self.adamsssim_loss_mri=AdaptiveMSSSIM(data_range=1.0, size_average=True, sets_winSize_weighting=self.MRI_sets_winSize_weighting)     
        self.adamsssim_loss_pet=AdaptiveMSSSIM(data_range=1.0, size_average=True, sets_winSize_weighting=self.PET_sets_winSize_weighting)     

    def _load_backbone(self):

        #===============================================
        # Load backbone model
        #===============================================
        if self.args.backbone==0:
            input_nc = 1
            output_nc = 1
            nb_filter = [64, 96, 128, 256]
            deepsupervision = False
            
            self.model_fusion = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=self.args.fusion, scale_head=4, fusion_op=True, classification=self.args.classification) 
        elif self.args.backbone==1:
            img_size=112
            patch_size=2#1
            in_chans=1
            out_chans=in_chans
            embed_dim=96
            depths=[2, 2, 2, 2]
            num_heads=[3, 6, 12, 24]
            window_size=7
            mlp_ratio=4.
            qkv_bias=True
            qk_scale=None
            drop_rate=0.
            drop_path_rate=0.1
            ape=True
            patch_norm=True
            use_checkpoint=False    
            self.model_fusion=SwinUNet(img_size=img_size,
                                patch_size=patch_size,
                                in_chans=in_chans,
                                num_classes=out_chans,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                ape=ape,
                                patch_norm=patch_norm,
                                use_checkpoint=use_checkpoint,
                                fusion_type=self.args.fusion,
                                fusion_op=True, 
                                classification=self.args.classification)
        #    
        if self.args.fusion_op == False:
            self.model_conversion = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, scale_head=4, fusion_op=False) 
        else:
            self.model_conversion=None

    def _load_weights(self):

        try:
            if self.args.resume_fusion_checkpoint:
                ckeckpoint_path=str(self.args.resume_fusion_checkpoint)
                if os.path.isfile(ckeckpoint_path):
                    print(torch.load(ckeckpoint_path).keys())
                    
                    self.model_fusion.load_state_dict(torch.load(ckeckpoint_path))
                else:
                    print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
        except AttributeError:
            print("AttributeError: 'Namespace' object has no attribute 'resume_fusion_checkpoint'")


        try:
            if self.args.resume_conversion_checkpoint:
                ckeckpoint_path=str(self.args.resume_conversion_checkpoint)
                if os.path.isfile(ckeckpoint_path):
                    print(torch.load(ckeckpoint_path).keys())
                    self.model_conversion.load_state_dict(torch.load(ckeckpoint_path))
                    
                else:
                    print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
        except AttributeError:
            print("AttributeError: 'Namespace' object has no attribute 'resume_conversion_checkpoint'")
    

    
    #move from utils to here 
    
    # def start_training_pipeline():
        
    # def _train():
        
    # def _save_images():
        
    # def _save_bestmodel():
        
    # def _valid():
        
    # tqdm...
    
    #integrate train_tune and train_cycle