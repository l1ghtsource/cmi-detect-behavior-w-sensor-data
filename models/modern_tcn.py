import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.modern_tcn_layers import series_decomp, Flatten_Head
from modules.revin import RevIN
from configs.config import cfg

# bad solo, bad in hybrid :(

class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(
            x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        x = self.dw(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):
    def __init__(self,task_name,patch_size,patch_stride, stem_ratio, downsample_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=cfg.seq_len, c_in=cfg.imu_vars, individual=False, target_window=96, class_drop=0.,class_num = 10):

        super(ModernTCN, self).__init__()

        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num


        # RevIN
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)


        # head
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage-1]


        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        else:
            if patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio,(self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)


            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.class_dropout = nn.Dropout(self.class_drop)

            self.head_class = nn.Linear(self.n_vars[0]*self.head_nf,self.class_num)


    def forward_feature(self, x, te=None):

        B,M,L=x.shape

        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:,:,-1:].repeat(1,1,pad_len)
                    x = torch.cat([x,pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def classification(self,x):

        x =  self.forward_feature(x,te=None)
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head_class(x)
        return x


    def forward(self, x, te=None):

        if self.task_name == 'classification':
            x = self.classification(x)

        return x



    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.task_name = configs.task_name
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        #classification
        self.class_dropout = configs.class_dropout
        self.class_num = configs.num_class


        # decomp
        self.decomposition = configs.decomposition


        self.model = ModernTCN(task_name=self.task_name,patch_size=self.patch_size, patch_stride=self.patch_stride, stem_ratio=self.stem_ratio,
                           downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,
                           large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                           nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                           backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                           use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                           subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in,
                           individual=self.individual, target_window=self.target_window,
                            class_drop = self.class_dropout, class_num = self.class_num)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.permute(0, 2, 1)
        te = None
        x = self.model(x, te)
        return x
    
class ModernTCN_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 task_name='classification',
                 patch_size=8,#3,
                 patch_stride=4,#1,
                 stem_ratio=1,
                 downsample_ratio=2,
                 ffn_ratio=1,#4,
                 num_blocks=[1, 1],#[2, 2],
                 large_size=[11, 11],
                 small_size=[3, 3],
                 dims=[48, 96],
                 dw_dims=[48, 96],
                 nvars=cfg.imu_vars,
                 small_kernel_merged=False,
                 backbone_dropout=0.3,
                 head_dropout=0.3,
                 use_multi_scale=True,
                 revin=True,
                 affine=True,
                 subtract_last=False,
                 seq_len=cfg.seq_len,
                 individual=False,
                 target_window=96,
                 class_drop=0.4,
                 class_num=18):
        super(ModernTCN_SingleSensor_v1, self).__init__()
        
        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num
        self.nvars = nvars
        self.seq_len = seq_len
        
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            stage = Stage(
                ffn_ratio=ffn_ratio,
                num_blocks=num_blocks[stage_idx],
                large_size=large_size[stage_idx],
                small_size=small_size[stage_idx],
                dmodel=dims[stage_idx],
                dw_model=dw_dims[stage_idx],
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout
            )
            self.stages.append(stage)
            
            if stage_idx < self.num_stage - 1:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[stage_idx]),
                    nn.Conv1d(dims[stage_idx], dims[stage_idx+1], 
                              kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)
        
        self.head_nf = dims[-1] * (seq_len // patch_stride) // (downsample_ratio ** (self.num_stage - 1))
        self.head = Flatten_Head(
            individual=individual,
            n_vars=nvars,
            nf=self.head_nf,
            target_window=target_window,
            head_dropout=head_dropout
        )
        
        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.class_dropout = nn.Dropout(class_drop)
            self.head_class1 = nn.Linear(nvars * self.head_nf, class_num)
            self.head_class2 = nn.Linear(nvars * self.head_nf, 2)

    def forward_feature(self, x):
        B, M, D, N = x.shape
        
        for i in range(self.num_stage):
            x = x.reshape(B * M, D, N)
            if i == 0:
                pad_len = self.downsample_layers[0][0].kernel_size[0] - self.downsample_layers[0][0].stride[0]
                x = F.pad(x, (0, pad_len), "constant", 0)
            x = self.downsample_layers[i](x)
            _, D, N = x.shape
            x = x.reshape(B, M, D, N)
            
            x = self.stages[i](x)
        return x

    def classification(self, x):
        if self.revin:
            x = x.permute(0, 2, 3, 1)  # [B, M, D, N] -> [B, D, N, M]
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 3, 1, 2)  # [B, M, D, N]
        
        x = self.forward_feature(x)
        
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = x.reshape(x.shape[0], -1)
        x1 = self.head_class1(x)
        x2 = self.head_class2(x)
        return x1, x2

    def forward(self, imu_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, seq_len, n_vars]
        """
        x = imu_data.permute(0, 3, 1, 2)  # [B, n_vars, 1, seq_len]
        
        if self.task_name == 'classification':
            return self.classification(x)
        
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

class SensorAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(input_dim // 2, 1)
    
    def forward(self, x):
        # x: [B, num_sensors, D]
        scores = self.fc2(self.act(self.fc1(x)))  # [B, num_sensors, 1]
        weights = F.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)  # [B, D]

class MultiSensor_ModernTCN_v1(nn.Module):
    def __init__(self, class_num=18):
        super().__init__()
        
        base_params = {
            'task_name': 'feature_extraction',
            'revin': False,
            'patch_size': 1,
            'patch_stride': 1,
            'downsample_ratio': 2,
            'ffn_ratio': 1,
            'small_kernel_merged': False,
            'backbone_dropout': 0.1
        }

        imu_params = {
            **base_params,
            'num_blocks': [1, 1],
            'large_size': [51, 51],
            'small_size': [5, 5],
            'dims': [62, 128],
            'dw_dims': [62, 128]
        }

        tof_params = {
            **base_params,
            'num_blocks': [1, 1],
            'large_size': [51, 51],
            'small_size': [5, 5],
            'dims': [62, 128],
            'dw_dims': [62, 128]
        }

        thm_params = {
            **base_params,
            'num_blocks': [1, 1],
            'large_size': [51, 51],
            'small_size': [5, 5],
            'dims': [62, 128],
            'dw_dims': [62, 128]
        }
        
        self.imu_branch = ModernTCN_SingleSensor_v1(
            in_channels=7,
            nvars=1,
            **imu_params
        )
        
        self.tof_branch = ModernTCN_SingleSensor_v1(
            in_channels=64,
            nvars=5,
            **tof_params
        )
        
        self.thm_branch = ModernTCN_SingleSensor_v1(
            in_channels=1,
            nvars=5,
            **thm_params
        )
        
        self.D_imu = imu_params['dims'][-1]
        self.D_tof = tof_params['dims'][-1]
        self.D_thm = thm_params['dims'][-1]
        
        self.tof_attn = SensorAttention(self.D_tof)
        self.thm_attn = SensorAttention(self.D_thm)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(self.D_imu + self.D_tof + self.D_thm, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, class_num)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.D_imu + self.D_tof + self.D_thm, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, imu_data, tof_data, thm_data, pad_mask=None):
        imu_data = imu_data.permute(0, 1, 3, 2)  # [B, 1, 7, L]
        imu_feat = self.imu_branch.forward_feature(imu_data)
        imu_feat = imu_feat.mean(dim=3)  # [B, 1, D_imu]
        imu_feat = imu_feat.squeeze(1)   # [B, D_imu]
        
        tof_data = tof_data.permute(0, 1, 3, 2)  # [B, 5, 64, L]
        tof_feat = self.tof_branch.forward_feature(tof_data)
        tof_feat = tof_feat.mean(dim=3)  # [B, 5, D_tof]
        tof_feat = self.tof_attn(tof_feat)  # [B, D_tof]
        
        thm_data = thm_data.permute(0, 1, 3, 2)  # [B, 5, 1, L]
        thm_feat = self.thm_branch.forward_feature(thm_data)
        thm_feat = thm_feat.mean(dim=3)  # [B, 5, D_thm]
        thm_feat = self.thm_attn(thm_feat)  # [B, D_thm]
        
        combined = torch.cat([
            imu_feat.unsqueeze(1), 
            tof_feat.unsqueeze(1), 
            thm_feat.unsqueeze(1)
        ], dim=1)  # [B, 3, D_imu+D_tof+D_thm]
        
        attn_output, _ = self.cross_attn(
            query=combined,
            key=combined,
            value=combined
        )
        attn_output = attn_output.mean(dim=1)  # [B, D_imu+D_tof+D_thm]
        x1 = self.classifier1(attn_output)
        x2 = self.classifier2(attn_output)
        
        return x1, x2