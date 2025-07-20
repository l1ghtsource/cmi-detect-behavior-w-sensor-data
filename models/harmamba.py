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
import random

from mamba_ssm.modules.mamba_simple import Mamba

from modules.rope import *
from modules.revin import RevIN

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn  # Legacy mambav1 file structure
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn  # mambav2 file structure
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from configs.config import cfg

# bad solo, bad in hybrid :(

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

class HARMamba(nn.Module):
    def __init__(self, 
                 seq_size=512,
                 patch_size=16,
                 stride=16,
                 depth=12,
                 embed_dim=64,
                 channels=1,
                 num_classes=12,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=True,
                 final_pool_type='mean',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=True,
                 bimamba_type="none",
                 if_cls_token=True,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 revin=True,
                 affine=True,
                 c_in=9,
                 subtract_last=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0


        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.patch_embed = PatchEmbed(
            seq_size=seq_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches * c_in

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = seq_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(self.num_features, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
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
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head1.apply(segm_init_weights)
        self.head2.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # print(f"x.device: {x.device}, pos_embed.device: {self.pos_embed.device}")

        # norm
        if self.revin:
            x = np.squeeze(x)
            x = self.revin_layer(x, 'norm')
            # denorm
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        x = x.unsqueeze(1)
        x = self.patch_embed(x)
        # print(x.shape)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    # M = x.shape[1]
                    token_position = 0
                    # x = torch.cat((x,cls_token), dim=1)

                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:

            # pos_embed = self.pos_embed
            # print(x.shape)
            # print(pos_embed.shape)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:


            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)


            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):

                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:

                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)




        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None

        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
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

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]
                    # return hidden_states[:, token_position, :],hidden_states[:,token_position,:]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
            # return hidden_states[:, -1, :],hidden_states[:,-1,:]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)

        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        x1 = self.head1(x)
        x2 = self.head2(x)
        if self.final_pool_type == 'max':
            x1 = x1.max(dim=1)[0]
            x2 = x2.max(dim=1)[0]
        return x1, x2

class HARMamba_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 seq_size=cfg.seq_len,
                 patch_size=16,
                 stride=8,
                 depth=6,
                 embed_dim=128,
                 num_classes=18,
                 **kwargs):
        super().__init__()
        
        self.harmamba = HARMamba(
            seq_size=seq_size,
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            channels=1,
            num_classes=num_classes,
            c_in=cfg.imu_vars,
            revin=True,
            **kwargs
        )
    
    def forward(self, imu_data, pad_mask=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            logits: [B, num_classes] - classification logits
        """
        x = imu_data.squeeze(1)  # [B, L, 7]
        out, out2 = self.harmamba(x)
        
        return out, out2
    
class MultiSensor_HARMamba_v1(nn.Module):
    def __init__(self, 
                 seq_size=cfg.seq_len,
                 patch_size=16,
                 stride=16,
                 depth=12,
                 embed_dim=256,
                 num_classes=18,
                 fusion_type='concat',  # 'concat', 'add', 'attention'
                 **kwargs):
        super().__init__()
        
        self.imu_harmamba = HARMamba(
            seq_size=seq_size,
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            channels=1,
            num_classes=0,
            c_in=cfg.imu_vars,
            revin=True,
            **kwargs
        )
        
        self.tof_harmamba = HARMamba(
            seq_size=seq_size,
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            channels=5,
            num_classes=0,
            c_in=64,
            revin=True,
            **kwargs
        )
        
        self.thm_harmamba = HARMamba(
            seq_size=seq_size,
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            channels=5,
            num_classes=0,
            c_in=1,
            revin=True,
            **kwargs
        )
        
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.classifier1 = nn.Linear(embed_dim * 3, num_classes)
            self.classifier2 = nn.Linear(embed_dim * 3, 2)
            self.classifier3 = nn.Linear(embed_dim * 3, 4)
        elif fusion_type == 'add':
            self.classifier1 = nn.Linear(embed_dim, num_classes)
            self.classifier2 = nn.Linear(embed_dim, 2)
            self.classifier3 = nn.Linear(embed_dim, 4)
        elif fusion_type == 'attention':
            self.attention_fusion = MultiModalAttentionFusion(embed_dim, num_modalities=3)
            self.classifier1 = nn.Linear(embed_dim, num_classes)
            self.classifier2 = nn.Linear(embed_dim, 2)
            self.classifier3 = nn.Linear(embed_dim, 4)
    
    def forward(self, imu_data, thm_data, tof_data, pad_mask=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        """
        # IMU: [B, 1, L, 7] -> [B, L, 7]
        imu_features = self.imu_harmamba(imu_data.squeeze(1), return_features=True)
        
        # ToF: [B, 5, L, 64] -> [B, L, 320] (5*64)
        B, n_sensors, L, n_vars = tof_data.shape
        tof_reshaped = tof_data.permute(0, 2, 1, 3).reshape(B, L, n_sensors * n_vars)
        tof_features = self.tof_harmamba(tof_reshaped, return_features=True)
        
        # THM: [B, 5, L, 1] -> [B, L, 5] (5*1)
        B, n_sensors, L, n_vars = thm_data.shape
        thm_reshaped = thm_data.permute(0, 2, 1, 3).reshape(B, L, n_sensors * n_vars)
        thm_features = self.thm_harmamba(thm_reshaped, return_features=True)
        
        if self.fusion_type == 'concat':
            fused_features = torch.cat([imu_features, tof_features, thm_features], dim=1)
        elif self.fusion_type == 'add':
            fused_features = imu_features + tof_features + thm_features
        elif self.fusion_type == 'attention':
            fused_features = self.attention_fusion([imu_features, tof_features, thm_features])
        
        logits1 = self.classifier1(fused_features)
        logits2 = self.classifier2(fused_features)
        logits3 = self.classifier3(fused_features)

        return logits1, logits2, logits3

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, modality_features):
        """
        Args:
            modality_features: [B, embed_dim]
        """
        stacked = torch.stack(modality_features, dim=1)
        
        attended, _ = self.attention(stacked, stacked, stacked)
        
        fused = attended.mean(dim=1)  # [B, embed_dim]
        fused = self.norm(fused)
        
        return fused