"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.nn import functional as F
import torch
import torch.nn as nn

import math

class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, feature_size, num_pos_feats=128, temperature=10000, normalize=False, scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Extraction_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ratio = 0.75
        
    def generate_random_mask(self, x, mask_ratio=0.75):
        N, L, D = x.shape  # Length = H*W
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, ids_keep

    def forward(self, x, y):
        B, T, C = x.shape
        _, N, _ = y.shape
        if self.training:
            y_masked, _ = self.generate_random_mask(y, 0.75)
            current_N = y_masked.shape[1] 
        else:
            y_masked = y
            current_N = N
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y_masked).reshape(B, current_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PPE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Extraction_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y, return_attention=False):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
   
        return x

class Structure_aware_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., windowsize=28, num_prototype=6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim *2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.windowsize = windowsize
        self.pos_embed = PositionEmbeddingSine([windowsize,windowsize], dim // 2, normalize=True)
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, dim))

        self.proto_centers = nn.Parameter(torch.rand(1, 1, num_prototype, 2))
        self.proto_log_sigma = nn.Parameter(torch.zeros(1, 1, num_prototype, 2))

        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def get_grid(self, N, device):
        H = int(N ** 0.5)
        W = N // H
        y_embed = torch.arange(H, dtype=torch.float32, device=device) / (H - 1 + 1e-6)
        x_embed = torch.arange(W, dtype=torch.float32, device=device) / (W - 1 + 1e-6)
        y_grid, x_grid = torch.meshgrid(y_embed, x_embed, indexing='ij')
        grid = torch.stack([x_grid, y_grid], dim=-1).flatten(0, 1) # [N, 2]
        return grid.unsqueeze(0) # [1, N, 2]
        
    def get_gradient_magnitude(self, x):
        B, N, C = x.shape
        H = int(N**0.5)
        feat = x.transpose(1, 2).contiguous().view(B, C, H, H)
        feat_intensity = feat.mean(dim=1, keepdim=True) # [B, 1, H, W]

        gx = F.conv2d(feat_intensity, self.sobel_x, padding=1)
        gy = F.conv2d(feat_intensity, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6) # [B, 1, H, W]
        return grad_mag.flatten(1) # [B, N]

    def generate_structure_aware_mask(self, x, mask_ratio=0.75):
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))
        importance = self.get_gradient_magnitude(x) # [B, N]
        min_val = importance.min(dim=1, keepdim=True)[0]
        max_val = importance.max(dim=1, keepdim=True)[0]
        importance_norm = (importance - min_val) / (max_val - min_val + 1e-6)
        noise = torch.rand_like(importance_norm)
        alpha = 0.5 # 0.5
        final_score = importance_norm + alpha * noise 
        _, ids_keep = torch.topk(final_score, k=len_keep, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
        return x_masked, ids_keep
        
    def forward(self, x, prototype_token):

        B, N, C = x.shape
        pos = self.pos_embed(x)
        prototype_num = prototype_token.shape[1]
        if self.training:
            x_masked, ids_keep = self.generate_structure_aware_mask(x, 0.5)
            decoder_q = self.mask_tokens.repeat(B, N, 1)
            decoder_q = torch.scatter(decoder_q, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C), x_masked)
            decoder_q = self.with_pos_embed(decoder_q, pos)
        else:
            decoder_q = self.with_pos_embed(x, pos)
        q = self.q(decoder_q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale

        grid = self.get_grid(N, device=x.device).unsqueeze(2)
        centers = self.proto_centers
        sigma = self.proto_log_sigma.exp()
        dist_sq = (grid - centers).pow(2)
        spatial_bias = - dist_sq / (2 * sigma**2 + 1e-6)
        spatial_bias = spatial_bias.sum(dim=-1)
        spatial_bias = spatial_bias.unsqueeze(1)
        attn = (attn + spatial_bias)

        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Texture_mapping_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim *2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, prototype_token):
        B, N, C = x.shape
        prototype_num = prototype_token.shape[1]
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STR(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_p = norm_layer(dim)
        self.attn1 = Texture_mapping_attention(
             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Structure_aware_attention(
             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, windowsize=28, num_prototype=6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    
    def forward(self, x, prototype, return_attention=False):
        
        x1 = self.attn1(self.norm1(x), self.norm1(prototype))
        x2 = self.attn2(self.norm2(x1), self.norm2(prototype)) 
        x = x1 + x2
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x
    