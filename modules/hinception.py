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
        previous_channels = in_channels  # For residual tracking
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
                    nn.Conv1d(previous_channels, current_channels, kernel_size=1, padding='same', bias=False),
                    nn.BatchNorm1d(current_channels)
                )
                self.shortcuts.append(shortcut)
                previous_channels = current_channels
        
    def _build_inception_module(self, input_channels, use_hybrid_layer=False):
        return InceptionModule(
            input_channels=input_channels,
            n_filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            bottleneck_size=self.bottleneck_size,
            use_bottleneck=self.use_bottleneck,
            use_hybrid=use_hybrid_layer,
            increasing_kernels=self.increasing_trend_kernels,
            decreasing_kernels=self.decreasing_trend_kernels,
            peak_kernels=self.peak_kernels,
            in_channels=self.in_channels
        )

    def _hybrid_layer(self, x, increasing_kernels, decreasing_kernels, peak_kernels, in_channels):
        conv_list = []
        
        # Increasing detection filters
        for kernel_size in increasing_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 == 0] *= -1
            filter_ = filter_.permute(2, 1, 0)  # To (out_channels, in_channels, kernel_size)
            conv = F.conv1d(x, filter_.to(x.device), padding=kernel_size//2, bias=None)
            conv_list.append(conv)
        
        # Decreasing detection filters
        for kernel_size in decreasing_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 > 0] *= -1
            filter_ = filter_.permute(2, 1, 0)
            conv = F.conv1d(x, filter_.to(x.device), padding=kernel_size//2, bias=None)
            conv_list.append(conv)
        
        # Peak detection filters
        for kernel_size in peak_kernels:
            length = kernel_size + kernel_size // 2
            filter_ = np.zeros((length, in_channels, 1))
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
        input_res = x
        shortcut_idx = 0
        for d, module in enumerate(self.inception_modules):
            out = module(x)
            
            # Residual connection every 3 modules
            if self.use_residual and d % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](input_res)
                out = F.relu(out + shortcut)
                input_res = out
                shortcut_idx += 1
            
            x = out  # Update for next module
        
        return x  # (bs, hidden, l)

class InceptionModule(nn.Module):
    def __init__(self, input_channels, n_filters, kernel_sizes, bottleneck_size, use_bottleneck, 
                 use_hybrid, increasing_kernels, decreasing_kernels, peak_kernels, in_channels):
        super(InceptionModule, self).__init__()
        
        self.use_bottleneck = use_bottleneck and input_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(input_channels, bottleneck_size, kernel_size=1, padding=0, bias=False)  # k=1 preserves length
            inception_channels = bottleneck_size
        else:
            inception_channels = input_channels
        
        # Inception convolutions with manual padding
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) // 2
            conv = nn.Conv1d(inception_channels, n_filters, kernel_size=k, padding=padding, bias=False)
            self.convs.append(conv)
        
        # Max pooling branch
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_after_pool = nn.Conv1d(input_channels, n_filters, kernel_size=1, padding=0, bias=False)
        
        # Hybrid params (unchanged)
        self.use_hybrid = use_hybrid
        self.increasing_kernels = increasing_kernels
        self.decreasing_kernels = decreasing_kernels
        self.peak_kernels = peak_kernels
        self.in_channels = in_channels
        
        # BatchNorm (unchanged)
        num_conv_filters = len(kernel_sizes) + 1
        num_hybrid_filters = len(increasing_kernels) + len(decreasing_kernels) + len(peak_kernels) if use_hybrid else 0
        self.bn = nn.BatchNorm1d(n_filters * num_conv_filters + num_hybrid_filters)
    
    def forward(self, x):
        original_length = x.shape[2]  # Track input length
        
        if self.use_bottleneck:
            input_inception = self.bottleneck(x)
        else:
            input_inception = x
        
        # Convs (apply match_length to each)
        convs_out = []
        for conv in self.convs:
            conv_out = conv(input_inception)
            conv_out = self.match_length(conv_out, original_length)
            convs_out.append(conv_out)
        
        # Max pool branch (apply match_length)
        max_pool_out = self.max_pool(x)
        conv_after = self.conv_after_pool(max_pool_out)
        conv_after = self.match_length(conv_after, original_length)
        convs_out.append(conv_after)
        
        # Hybrid layer if enabled (apply match_length)
        if self.use_hybrid:
            hybrid_out = self._hybrid_layer(x, self.increasing_kernels, self.decreasing_kernels, 
                                            self.peak_kernels, self.in_channels, original_length)
            convs_out.append(hybrid_out)
        
        # Concat and activate
        out = torch.cat(convs_out, dim=1)
        out = self.bn(out)
        out = F.relu(out)
        return out
    
    def match_length(self, tensor, target_len):
        current_len = tensor.shape[2]
        if current_len > target_len:
            return tensor[:, :, :target_len]  # Crop excess
        elif current_len < target_len:
            pad = target_len - current_len
            return F.pad(tensor, (0, pad))  # Pad on the right
        return tensor
    
    def _hybrid_layer(self, x, increasing_kernels, decreasing_kernels, peak_kernels, in_channels, target_length):
        conv_list = []
        
        # Increasing detection filters
        for kernel_size in increasing_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 == 0] *= -1
            filter_ = filter_.permute(2, 1, 0)
            padding = (kernel_size - 1) // 2
            conv = F.conv1d(x, filter_.to(x.device), padding=padding, bias=None)
            conv = self.match_length(conv, target_length)
            conv_list.append(conv)
        
        # Decreasing detection filters
        for kernel_size in decreasing_kernels:
            filter_ = torch.from_numpy(np.ones((kernel_size, in_channels, 1))).float()
            indices = np.arange(kernel_size)
            filter_[indices % 2 > 0] *= -1
            filter_ = filter_.permute(2, 1, 0)
            padding = (kernel_size - 1) // 2
            conv = F.conv1d(x, filter_.to(x.device), padding=padding, bias=None)
            conv = self.match_length(conv, target_length)
            conv_list.append(conv)
        
        # Peak detection filters
        for kernel_size in peak_kernels:
            length = kernel_size + kernel_size // 2
            filter_ = np.zeros((length, in_channels, 1))
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
            padding = (length - 1) // 2
            conv = F.conv1d(x, filter_.to(x.device), padding=padding, bias=None)
            conv = self.match_length(conv, target_length)
            conv_list.append(conv)
        
        hybrid = torch.cat(conv_list, dim=1)
        return F.relu(hybrid)