import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# take idea from https://github.com/MSD-IRIMAS/CF-4-TSC/blob/main/classifiers/H_Inception.py

class HInceptionTimeFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, length_TS=100, n_filters=32, use_residual=True, 
                 use_bottleneck=True, depth=6, max_cf_length=6):
        super(HInceptionTimeFeatureExtractor, self).__init__()
        
        self.in_channels = in_channels
        self.length_TS = length_TS
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_sizes = [40, 20, 10]
        self.bottleneck_size = n_filters
        self.max_cf_length = max_cf_length
        
        # Kernel sizes for custom filters
        self.increasing_trend_kernels = [2**i for i in range(1, self.max_cf_length + 1)]
        self.decreasing_trend_kernels = [2**i for i in range(1, self.max_cf_length + 1)]
        self.peak_kernels = [2**i for i in range(2, self.max_cf_length + 1)]
        
        # Build inception modules as a list
        self.inception_modules = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        current_channels = in_channels
        for d in range(depth):
            module = self._build_inception_module(current_channels, use_hybrid_layer=(d == 0))
            self.inception_modules.append(module)
            
            # Update channels after inception module
            num_conv_filters = len(self.kernel_sizes) + 1  # convs + maxpool conv
            if d == 0:
                num_hybrid_filters = len(self.increasing_trend_kernels) + len(self.decreasing_trend_kernels) + len(self.peak_kernels)
                current_channels = self.n_filters * num_conv_filters + num_hybrid_filters
            else:
                current_channels = self.n_filters * num_conv_filters
            
            # Add shortcut layer for residual every 3 modules
            if use_residual and d % 3 == 2:
                shortcut = nn.Sequential(
                    nn.Conv1d(in_channels if d == 2 else previous_channels, current_channels, kernel_size=1, padding='same', bias=False),
                    nn.BatchNorm1d(current_channels)
                )
                self.shortcuts.append(shortcut)
                previous_channels = current_channels  # Update for next residual
        
    def _build_inception_module(self, input_channels, use_hybrid_layer=False):
        layers = []
        
        # Bottleneck if enabled and input channels > 1
        if self.use_bottleneck and input_channels > 1:
            bottleneck = nn.Conv1d(input_channels, self.bottleneck_size, kernel_size=1, padding='same', bias=False)
            layers.append(bottleneck)
            inception_channels = self.bottleneck_size
        else:
            inception_channels = input_channels
        
        # Inception convolutions
        convs = []
        for k in self.kernel_sizes:
            conv = nn.Conv1d(inception_channels, self.n_filters, kernel_size=k, padding='same', bias=False)
            convs.append(conv)
        
        # Max pooling branch
        max_pool = nn.MaxPool1d(pool_size=3, stride=1, padding=1)
        conv_after_pool = nn.Conv1d(input_channels, self.n_filters, kernel_size=1, padding='same', bias=False)
        
        # Store for forward pass
        return {
            'bottleneck': layers[0] if layers else None,
            'convs': nn.ModuleList(convs),
            'max_pool': max_pool,
            'conv_after_pool': conv_after_pool,
            'use_hybrid': use_hybrid_layer,
            'bn': nn.BatchNorm1d(self.n_filters * (len(self.kernel_sizes) + 1) + (len(self.increasing_trend_kernels) * 3 - len(self.increasing_trend_kernels) if use_hybrid_layer else 0))
        }
    
    def _hybrid_layer(self, x):
        conv_list = []
        
        # Increasing detection filters
        for kernel_size in self.increasing_trend_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, self.in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 == 0] *= -1
            filter_ = filter_.permute(2, 1, 0)  # To (out_channels, in_channels, kernel_size)
            conv = F.conv1d(x, filter_.to(x.device), padding=kernel_size//2, bias=None)
            conv_list.append(conv)
        
        # Decreasing detection filters
        for kernel_size in self.decreasing_trend_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, self.in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 > 0] *= -1
            filter_ = filter_.permute(2, 1, 0)
            conv = F.conv1d(x, filter_.to(x.device), padding=kernel_size//2, bias=None)
            conv_list.append(conv)
        
        # Peak detection filters
        for kernel_size in self.peak_kernels:
            length = kernel_size + kernel_size // 2
            filter_ = np.zeros((length, self.in_channels, 1))
            xmesh = np.linspace(0, 1, kernel_size//4 + 1)[1:].reshape(-1, 1, 1)
            filter_left = xmesh ** 2
            filter_right = filter_left[::-1]
            filter_[0:kernel_size//4] = -filter_left
            filter_[kernel_size//4:kernel_size//2] = -filter_right
            filter_[kernel_size//2:3*kernel_size//4] = 2 * filter_left
            filter_[3*kernel_size//4:kernel_size] = 2 * filter_right
            filter_[kernel_size:5*kernel_size//4] = -filter_left
            filter_[5*kernel_size//4:] = -filter_right
            filter_ = torch.from_numpy(filter_).float().permute(2, 1, 0)
            conv = F.conv1d(x, filter_.to(x.device), padding=length//2, bias=None)
            conv_list.append(conv)
        
        hybrid = torch.cat(conv_list, dim=1)
        return F.relu(hybrid)
    
    def forward(self, x):
        # x: (bs, c, l)
        input_res = x
        for d, module in enumerate(self.inception_modules):
            # Bottleneck
            if module['bottleneck'] is not None:
                input_inception = module['bottleneck'](x)
            else:
                input_inception = x
            
            # Convs
            convs_out = [conv(input_inception) for conv in module['convs']]
            
            # Max pool branch
            max_pool_out = module['max_pool'](x)
            conv_after = module['conv_after_pool'](max_pool_out)
            convs_out.append(conv_after)
            
            # Hybrid layer if enabled
            if module['use_hybrid']:
                hybrid_out = self._hybrid_layer(x)
                convs_out.append(hybrid_out)
            
            # Concat and activate
            out = torch.cat(convs_out, dim=1)
            out = module['bn'](out)
            out = F.relu(out)
            
            # Residual connection every 3 modules
            if self.use_residual and d % 3 == 2:
                shortcut = self.shortcuts[d // 3](input_res)
                out = F.relu(out + shortcut)
                input_res = out
            
            x = out  # Update for next module
        
        return out  # (bs, hidden, l)