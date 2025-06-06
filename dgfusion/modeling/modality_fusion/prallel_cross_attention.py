"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/HRFuser/blob/master/mmdet/models/backbones/hrfuser_hrformer_based.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowMCA(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 kdim=None,
                 vdim=None,
                 with_rpe=True,
                 with_qc=False,
                 with_depth_pos_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.Wh, self.Ww = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dim = embed_dim // num_heads
        self.scale = qk_scale or head_embed_dim**-0.5

        self.with_rpe = with_rpe
        self.with_qc = with_qc
        self.with_depth_pos_bias = with_depth_pos_bias
        if self.with_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.Wh - 1) * (2 * self.Ww - 1), num_heads))

            coords_h = torch.arange(self.Wh)
            coords_w = torch.arange(self.Ww)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.Wh - 1
            relative_coords[:, :, 1] += self.Ww - 1
            relative_coords[:, :, 0] *= 2 * self.Ww - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

            if self.with_qc:
                self.global_query_bias = nn.Parameter(torch.zeros(num_heads, 1,1))

            if self.with_depth_pos_bias:
                self.global_depth_bias = nn.Parameter(torch.zeros(num_heads, 1,1))

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=qkv_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        B, Nq, C = query.shape 
        _, Nk, C = key.shape
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if self.with_rpe:
            window_dim = self.Wh * self.Ww
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                window_dim, window_dim, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

            extra_query = attn.shape[-2] - window_dim
            extra_key = attn.shape[-1] - window_dim

            # If extra tokens exist along the query dimension, add their biases
            if extra_query > 0:
                extra_bias = []
                if self.with_qc:
                    extra_bias.append(self.global_query_bias.expand(self.num_heads, 1, window_dim))
                if self.with_depth_pos_bias:
                    while len(extra_bias) < extra_query:
                        extra_bias.append(self.global_depth_bias.expand(self.num_heads, 1, window_dim))                
                extra_bias = torch.cat(extra_bias, dim=1)
                relative_position_bias = torch.cat((relative_position_bias, extra_bias), dim=1)

            # If extra tokens exist along the key dimension, add their biases
            if extra_key > 0:
                bias_extra_k = self.global_query_bias.expand(self.num_heads, relative_position_bias.shape[-2], extra_key)
                relative_position_bias = torch.cat((relative_position_bias, bias_extra_k), dim=2)

            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, Nq, Nk) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Nq, Nk)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class MultiWindowCrossAttention(nn.Module):
    def __init__(self,
                 window_size=7,
                 with_pad_mask=False,
                 **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMCA(window_size=self.window_size, **kwargs)

    def forward(self, x, y, H, W, **kwargs):
        assert x.shape == y.shape
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        Wh, Ww = self.window_size

        # center-pad the feature on H and W axes
        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = F.pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        y = F.pad(y, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # permute
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)
        y = y.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)

        # attention
        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = F.pad(pad_mask, [0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, y, y, pad_mask, **kwargs)
        else:
            out = self.attn(x, y, y, **kwargs)

        # reverse permutation
        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H + pad_h, W + pad_w, C)
        
        # de-pad
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)

class HRFuserFusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN'),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 main_modality=["CAMERA"],
                 secondary_modalities=["LIDAR", "EVENT_CAMERA", "RADAR", "REF_IMAGE"],
                 attend_to_x_tmp=True,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.main_modality = main_modality
        self.secondary_modalities = secondary_modalities
        self.attend_to_x_tmp = attend_to_x_tmp

        self.norm1 = nn.ModuleDict({modality: nn.LayerNorm(in_channels) for modality in self.secondary_modalities})
        self.norm2 = nn.ModuleDict({modality: nn.LayerNorm(out_channels) for modality in self.secondary_modalities})
        self.attn = nn.ModuleDict({modality: MultiWindowCrossAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            **kwargs) for modality in self.secondary_modalities})
        
        self.norm3 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(in_channels * mlp_ratio, out_channels),
        )
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, modality_features):
        B, C, H, W = modality_features[self.main_modality].size()
        x = modality_features[self.main_modality].view(B, -1, C)
        x_tmp = x.clone() if self.attend_to_x_tmp else None
        for modality in self.secondary_modalities:
            z = modality_features[modality].view(B, -1, C)
            query = x_tmp if self.attend_to_x_tmp else x
            x = x + z + self.drop_path(self.attn[modality](self.norm1[modality](query), self.norm2[modality](z), H, W))
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        x = x.view(B, C, H, W)
        return x

class ParallelCrossAttention(nn.Module):
    def __init__(self,
                 cfg,
                 modalities,
                 input_shapes,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 ):
        super().__init__()
        self.expects_q_condition = False
        self.modalities = modalities
        self.number_modalities = len(modalities)
        self.levels = cfg.MODEL.FUSION.LEVELS
        self.levels_map = {'res2':0, 'res3':1, 'res4':2, 'res5':3}
        self.fusion = self._make_multimodal_fusion(cfg, cfg.MODEL.FUSION.PCA, input_shapes, 
                                                    transformer_norm_cfg, norm_cfg)

    def forward(self, src):
        outputs = {}
        for level, features in src.items():
            if level in self.levels:
                outputs[level] = self.fusion[level](features)
            else:
                outputs[level] = features[self.modalities[0]]
        return outputs
        

    def _make_multimodal_fusion(self, cfg, layer_config, input_shapes, transformer_norm_cfg, norm_cfg):
        num_heads = layer_config.NHEAD
        num_window_size = layer_config.WINDOW_SIZE
        num_mlp_ratio = layer_config.MLP_RATIO
        drop_path = layer_config.DROP_PATH
        proj_drop_rate = layer_config.PROJ_DROP_RATE
        attn_drop_rate = layer_config.ATTN_DROP_RATE
        attend_to_x_tmp = cfg.MODEL.FUSION.PCA.ATTEND_TO_X_TMP
        with_pad_mask = layer_config.WITH_PAD_MASK

        fusion_modules = nn.ModuleDict()
        
        for i, level in enumerate(self.levels):
            fusion_modules[level] = HRFuserFusionBlock(
                input_shapes[level].channels,
                input_shapes[level].channels,
                num_heads=num_heads[self.levels_map[level]],
                window_size=num_window_size,
                mlp_ratio=num_mlp_ratio,
                drop_path=drop_path,
                norm_cfg=norm_cfg,
                transformer_norm_cfg=transformer_norm_cfg,
                main_modality=self.modalities[0],
                secondary_modalities=self.modalities[1:],
                proj_drop_rate=proj_drop_rate,
                attn_drop_rate=attn_drop_rate,
                attend_to_x_tmp=attend_to_x_tmp,
                with_pad_mask=with_pad_mask,
            )

        return fusion_modules
