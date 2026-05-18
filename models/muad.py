import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class STAR(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            extraction,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
    ) -> None:
        super(STAR, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.Extraction = extraction
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0


    def s_p_loss(self, features, inps):
        features_norm = F.normalize(features, p=2, dim=-1) # (B, N, C)
        inps_norm = F.normalize(inps, p=2, dim=-1)         # (B, M, C)
        sim_matrix = torch.bmm(features_norm, inps_norm.transpose(1, 2))
        weights = F.softmax(sim_matrix, dim=-1) # (B, N, M)
        reconstructed_features = torch.bmm(weights, inps)
        features_vec = features.view(features.size(0), -1)                # (B, N*C)
        self.distance = 1. - F.cosine_similarity(features, reconstructed_features, dim=-1)
        reconstructed_vec = reconstructed_features.view(features.size(0), -1) # (B, N*C)
        
        global_sim = F.cosine_similarity(features_vec, reconstructed_vec, dim=1)
        loss = 1 - global_sim
        return loss.mean()
    

    def forward(self, x):

        x = self.encoder.prepare_tokens(x)
        B, _, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
        
    
        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
        x = self.fuse_feature(en_list)

        learned_token = self.prototype_token.unsqueeze(0).repeat((B, 1, 1))

        for i, blk in enumerate(self.Extraction):
            p = blk(learned_token, x)
        sp_loss = 0
        sp_loss += self.s_p_loss(x, p)


        for i, blk in enumerate(self.bottleneck):
            x = blk(x)
        
        de_list = []

        for i, blk in enumerate(self.decoder):

            x = blk(x, p)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        
        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]
            
        
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

    

        return en, de, sp_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































