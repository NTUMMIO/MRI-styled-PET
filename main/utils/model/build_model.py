import os
import torch
import numpy as np
import torch.nn as nn
# from utils.model.backbone.unet import NestFuse
# from utils.model.backbone.swinunet import SwinUNet
from utils.model.backbone.backbone import UNetPP, SwinUNet
def build_model(args, fold=0):

    if args.backbone==0:
        
        input_nc = 1
        output_nc = 1
        #nb_filter = [64, 112, 160, 208, 256]
        nb_filter = [64, 96, 128, 256]
        deepsupervision = False
        
        # model_x2y = NestFuse(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=args.fusion, scale_head=4, fusion_op=True) 
        model_x2y = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=args.fusion, scale_head=4, fusion_op=True, classification=args.classification) 
        
        print("-------------currently using unet++ as backbone----------------------")
        if args.fusion_op == False:
            model_y2x = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=args.fusion, scale_head=4, fusion_op=False) 
        
    elif args.backbone==1:

        img_size=112
        patch_size=2#1
        in_chans=1
        out_chans=in_chans
        num_classes=1
        embed_dim=96
        depths=[2, 2, 2, 2]
        depths_decoder=[1, 2, 2, 2]
        num_heads=[3, 6, 12, 24]
        window_size=7
        mlp_ratio=4.
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.1
        norm_layer=nn.LayerNorm
        ape=True
        patch_norm=True
        use_checkpoint=False    
        model_x2y=SwinUNet(img_size=img_size,
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
                            fusion_type=args.fusion,
                            fusion_op=True, 
                            classification=args.classification)
        # if args.fusion_op == False:
        #     model_y2x=SwinUNet(img_size=img_size,
        #                         patch_size=patch_size,
        #                         in_chans=in_chans,
        #                         num_classes=out_chans,
        #                         embed_dim=embed_dim,
        #                         depths=depths,
        #                         num_heads=num_heads,
        #                         window_size=window_size,
        #                         mlp_ratio=mlp_ratio,
        #                         qkv_bias=qkv_bias,
        #                         qk_scale=qk_scale,
        #                         drop_rate=drop_rate,
        #                         drop_path_rate=drop_path_rate,
        #                         ape=ape,
        #                         patch_norm=patch_norm,
        #                         use_checkpoint=use_checkpoint,
        #                         fusion_op=False)
        print("-------------currently using swinunet as backbone----------------------")
    
    # ==========================
    # number of parameters
    para = sum([np.prod(list(p.size())) for p in model_x2y.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model_x2y._get_name(), para * type_size / 1000 / 1000))
    # ==========================
    # resume from checkpoint
    try:
        if args.resume_x2y:
            ckeckpoint_path=str(args.resume_x2y[fold])
            if os.path.isfile(ckeckpoint_path):
                print(torch.load(ckeckpoint_path).keys())
                
                # Because fusion=1(sc-att) has no trainable parameters
                # if args.fusion_type==1:
                #     checkpoint=torch.load(ckeckpoint_path)
                #     keys_to_remove = ["fusion0.weight", "fusion1.weight", "fusion2.weight", "fusion3.weight"]

                #     # Remove the specified keys from the state_dict
                #     filtered_state_dict = {key: value for key, value in checkpoint['state_dict'].items() if key not in keys_to_remove}

                #     model_x2y.load_state_dict(filtered_state_dict)
                # else:
                model_x2y.load_state_dict(torch.load(ckeckpoint_path))
            else:
                print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        print("AttributeError: 'Namespace' object has no attribute 'resume_x2y'")
    try:
        if args.resume_y2x:
            ckeckpoint_path=str(args.resume_y2x[fold])
            if os.path.isfile(ckeckpoint_path):
                print(torch.load(ckeckpoint_path).keys())
                model_y2x.load_state_dict(torch.load(ckeckpoint_path))
                
            else:
                print("=> no checkpoint found at '{}'".format(ckeckpoint_path))
    except AttributeError:
        print("AttributeError: 'Namespace' object has no attribute 'resume_y2x'")
    # ==========================
    if args.fusion_op:            
        return model_x2y
    else:
        input_nc = 1
        output_nc = 1
        #nb_filter = [64, 112, 160, 208, 256]
        nb_filter = [64, 96, 128, 256]
        deepsupervision = False
        # model_y2x = NestFuse(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=args.fusion, scale_head=4, fusion_op=False) 
        model_y2x = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, scale_head=4, fusion_op=False) 
        
        return model_x2y, model_y2x

