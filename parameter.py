import torch
import torch.nn as nn
import os
from functools import partial
import warnings
import argparse
from utils import setup_seed
from thop import profile
# Model-Related Modules
from models import vit_encoder
from models.muad import STAR
from models.vision_transformer import Mlp, PPE, STR
from flops_profiler.profiler import get_model_profile



warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info

    if 'dinov2' in args.encoder:
        encoder = vit_encoder.load(args.encoder)
        if 'small' in args.encoder:
            embed_dim, num_heads = 384, 6
        elif 'base' in args.encoder:
            embed_dim, num_heads = 768, 12
        elif 'large' in args.encoder:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."

    elif 'dinov3' in args.encoder:
        if 'l' in args.encoder:
            repo_dir = './dinov3'
            Dinov3_model_path = '/userHome/why/ChenjieFiles/AD-DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
            encoder = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=Dinov3_model_path)
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

        elif 'b' in args.encoder:
            repo_dir = './dinov3'
            Dinov3_model_path = '/userHome/why/ChenjieFiles/AD-DINOv3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            encoder = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=Dinov3_model_path)
            embed_dim, num_heads = 768, 12
        
        elif 's' in args.encoder:
            repo_dir = './dinov3'
            Dinov3_model_path = '/userHome/why/ChenjieFiles/INP-Former/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
            encoder = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=Dinov3_model_path)
            embed_dim, num_heads = 384, 6

# Model Preparation
    Bottleneck = []
    R = []
    P_Extractor = []
    
    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    P = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.P_num, embed_dim))
                     for _ in range(1)])
    for i in range(1):
        blk = PPE(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        P_Extractor.append(blk)
    P_Extractor = nn.ModuleList(P_Extractor)

    for i in range(8):
        blk = STR(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        R.append(blk)
    R = nn.ModuleList(R)
  

    model = STAR(encoder=encoder, bottleneck=Bottleneck, 
                       extraction=P_Extractor,
                       decoder=R, 
                       target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                       fuse_layer_decoder=fuse_layer_decoder, prototype_token=P,
                       )
    model = model.to(device)
  

    model = model.to(device)
    dummy_input = torch.randn(1, 3, 280, 280, dtype=torch.float).to(device)
    model.eval()
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')
    

    
    # model info        
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov3_vitb16' or 'dinov3_vitl16 or dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448) # '448, 512'
    parser.add_argument('--crop_size', type=int, default=224) # '392, 448'
    parser.add_argument('--INP_num', type=int, default=6)


    args = parser.parse_args()


    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    main(args)
