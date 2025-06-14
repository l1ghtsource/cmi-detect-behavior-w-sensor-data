# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import numpy as np
import torch
import torch.nn as nn
from functools import partial

from torch import Tensor
from typing import Optional

from modules.weight_init import trunc_normal_, lecun_normal_
from modules.drop import DropPath
from modules.helpers import to_2tuple

import math

from mamba_ssm.modules.mamba_simple import Mamba

from modules.rope import *
from modules.revin import RevIN

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from configs.config import cfg

# TODO: test it

class PatchEmbed(nn.Module):
    """ Patch Embedding
    """
    def __init__(self, seq_size=None, patch_size=None, stride=None, in_chans=1, embed_dim=64, norm_layer=None, flatten=True):
        super().__init__()
        seq_size = to_2tuple(seq_size)
        kersize = patch_size
        ssieze = stride
        patch_size = to_2tuple(patch_size)
        self.seq_size = seq_size
        self.patch_size = patch_size
        self.grid_size = ((seq_size[0] - patch_size[0]) // stride + 1, (seq_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = (seq_size[0] - patch_size[0]) // stride + 1
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(kersize,1),
            stride=(ssieze,1),
            padding=0
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):


        x = self.proj(x)
        # print(x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type=None,
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx,
                        # bimamba_type=bimamba_type,
                        # if_devide_out=if_devide_out,
                        # init_layer_scale=init_layer_scale,
                        **ssm_cfg,
                        **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class HARMamba_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 seq_size=cfg.seq_len,
                 patch_size=16,
                 stride=16,
                 depth=12,
                 embed_dim=64,
                 num_classes=cfg.main_num_classes,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon=1e-5, 
                 rms_norm=False, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=True,
                 final_pool_type='mean',
                 if_abs_pos_embed=True,
                 if_bimamba=True,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 revin=True,
                 affine=True,
                 subtract_last=False,
                 **kwargs):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        
        # RevIN for normalization
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(7, affine=affine, subtract_last=subtract_last)  # 7 IMU channels
        
        # Patch embedding IMU [B, 1, L, 7] -> [B, num_patches, embed_dim]
        self.patch_embed = PatchEmbed(
            seq_size=seq_size, 
            patch_size=patch_size, 
            stride=stride, 
            in_chans=7,  # 7 IMU channels 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Mamba layers
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])
        
        # Output normalization and classification head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        self.head = nn.Linear(self.num_features, num_classes)
        
        # Initialize weights
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, imu_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        # Remove singleton dimension: [B, 1, L, 7] -> [B, L, 7]
        x = imu_data.squeeze(1)
        B, L, C = x.shape
        
        # Apply RevIN normalization if enabled
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        # Patch embedding: [B, L, 7] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x.unsqueeze(1))  # Add channel dim for patch_embed
        
        # Add positional embedding
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)
        
        # Process through Mamba layers
        residual = None
        hidden_states = x
        
        if not self.if_bidirectional:
            # Unidirectional processing
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            # Bidirectional processing
            for i in range(len(self.layers) // 2):
                hidden_states_f, residual_f = self.layers[i * 2](hidden_states, residual)
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), 
                    None if residual is None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        
        # Final normalization
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # Global pooling
        if self.final_pool_type == 'mean':
            features = hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            features = hidden_states.max(dim=1)[0]
        else:  # 'none' - use last token
            features = hidden_states[:, -1, :]
        
        # Apply RevIN denormalization if enabled
        if self.revin:
            features = self.revin_layer(features, 'denorm')
        
        # Classification
        logits = self.head(features)
        
        return logits
    
class MultiSensor_HARMamba_v1(nn.Module):
    def __init__(self, 
                 seq_size=cfg.seq_len,
                 patch_size=16,
                 stride=16,
                 depth=12,
                 embed_dim=64,
                 num_classes=cfg.main_num_classes,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon=1e-5, 
                 rms_norm=False, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=True,
                 final_pool_type='mean',
                 if_abs_pos_embed=True,
                 if_bimamba=True,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 revin=True,
                 affine=True,
                 subtract_last=False,
                 fusion_strategy='concat',  # 'concat', 'add', 'attention'
                 **kwargs):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.d_model = self.num_features = self.embed_dim = embed_dim
        
        # RevIN layers for each sensor modality
        self.revin = revin
        if self.revin:
            self.revin_imu = RevIN(7, affine=affine, subtract_last=subtract_last)
            self.revin_tof = RevIN(64, affine=affine, subtract_last=subtract_last) 
            self.revin_thm = RevIN(1, affine=affine, subtract_last=subtract_last)
        
        # Separate patch embeddings for each sensor type
        self.patch_embed_imu = PatchEmbed(
            seq_size=seq_size, 
            patch_size=patch_size, 
            stride=stride, 
            in_chans=7,  # IMU: 7 channels
            embed_dim=embed_dim
        )
        
        self.patch_embed_tof = PatchEmbed(
            seq_size=seq_size, 
            patch_size=patch_size, 
            stride=stride, 
            in_chans=64,  # ToF: 64 channels
            embed_dim=embed_dim
        )
        
        self.patch_embed_thm = PatchEmbed(
            seq_size=seq_size, 
            patch_size=patch_size, 
            stride=stride, 
            in_chans=1,  # Thermal: 1 channel
            embed_dim=embed_dim
        )
        
        # Calculate total number of patches from all sensors
        num_patches_imu = self.patch_embed_imu.num_patches
        num_patches_tof = self.patch_embed_tof.num_patches * 5  # 5 ToF sensors
        num_patches_thm = self.patch_embed_thm.num_patches * 5  # 5 Thermal sensors
        
        if fusion_strategy == 'concat':
            total_patches = num_patches_imu + num_patches_tof + num_patches_thm
        else:  # 'add' or 'attention'
            total_patches = max(num_patches_imu, num_patches_tof, num_patches_thm)
        
        # Sensor type embeddings to distinguish different modalities
        self.sensor_type_embed = nn.Parameter(torch.zeros(3, embed_dim))  # 3 sensor types
        
        # Positional embeddings
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Attention-based fusion module if needed
        if fusion_strategy == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim, num_heads=8, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Mamba layers
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])
        
        # Output normalization and classification head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.patch_embed_imu.apply(segm_init_weights)
        self.patch_embed_tof.apply(segm_init_weights)
        self.patch_embed_thm.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.sensor_type_embed, std=.02)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, imu_data, thm_data, tof_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        B = imu_data.shape[0]
        
        # Process IMU data
        imu_x = imu_data.squeeze(1)  # [B, L, 7]
        if self.revin:
            imu_x = self.revin_imu(imu_x, 'norm')
        imu_patches = self.patch_embed_imu(imu_x.unsqueeze(1))  # [B, num_patches_imu, embed_dim]
        imu_patches = imu_patches + self.sensor_type_embed[0]  # Add sensor type embedding
        
        # Process ToF data (5 sensors)
        tof_patches_list = []
        for i in range(5):
            tof_x = tof_data[:, i, :, :]  # [B, L, 64]
            if self.revin:
                tof_x = self.revin_tof(tof_x, 'norm')
            tof_patch = self.patch_embed_tof(tof_x.unsqueeze(1))  # [B, num_patches_tof, embed_dim]
            tof_patch = tof_patch + self.sensor_type_embed[1]  # Add sensor type embedding
            tof_patches_list.append(tof_patch)
        tof_patches = torch.cat(tof_patches_list, dim=1)  # [B, 5*num_patches_tof, embed_dim]
        
        # Process Thermal data (5 sensors)  
        thm_patches_list = []
        for i in range(5):
            thm_x = thm_data[:, i, :, :]  # [B, L, 1]
            if self.revin:
                thm_x = self.revin_thm(thm_x, 'norm')
            thm_patch = self.patch_embed_thm(thm_x.unsqueeze(1))  # [B, num_patches_thm, embed_dim]
            thm_patch = thm_patch + self.sensor_type_embed[2]  # Add sensor type embedding
            thm_patches_list.append(thm_patch)
        thm_patches = torch.cat(thm_patches_list, dim=1)  # [B, 5*num_patches_thm, embed_dim]
        
        # Sensor fusion
        if self.fusion_strategy == 'concat':
            # Concatenate all patches
            x = torch.cat([imu_patches, tof_patches, thm_patches], dim=1)
        elif self.fusion_strategy == 'add':
            # Add patches (requires same number of patches)
            min_patches = min(imu_patches.shape[1], tof_patches.shape[1], thm_patches.shape[1])
            x = (imu_patches[:, :min_patches, :] + 
                 tof_patches[:, :min_patches, :] + 
                 thm_patches[:, :min_patches, :]) / 3
        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            all_patches = torch.cat([imu_patches, tof_patches, thm_patches], dim=1)
            fused_patches, _ = self.fusion_attention(all_patches, all_patches, all_patches)
            x = self.fusion_norm(fused_patches + all_patches)
        
        # Add positional embeddings
        if self.if_abs_pos_embed:
            if x.shape[1] <= self.pos_embed.shape[1]:
                x = x + self.pos_embed[:, :x.shape[1], :]
            else:
                # Interpolate positional embeddings if needed
                pos_embed_interp = torch.nn.functional.interpolate(
                    self.pos_embed.transpose(1, 2), 
                    size=x.shape[1], 
                    mode='linear'
                ).transpose(1, 2)
                x = x + pos_embed_interp
            x = self.pos_drop(x)
        
        # Process through Mamba layers
        residual = None
        hidden_states = x
        
        if not self.if_bidirectional:
            # Unidirectional processing
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            # Bidirectional processing
            for i in range(len(self.layers) // 2):
                hidden_states_f, residual_f = self.layers[i * 2](hidden_states, residual)
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), 
                    None if residual is None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        
        # Final normalization
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # Global pooling
        if self.final_pool_type == 'mean':
            features = hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            features = hidden_states.max(dim=1)[0]
        else:  # 'none' - use last token
            features = hidden_states[:, -1, :]
        
        # Apply RevIN denormalization if needed
        if self.revin:
            # For features, we can apply any of the revin layers for denorm
            features = self.revin_imu(features, 'denorm')
        
        # Classification
        logits = self.head(features)
        
        return logits