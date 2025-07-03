import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/MSD-IRIMAS/LITE/blob/main/classifiers/lite.py <-- I HATE TENSORFLOW !!!!

def manual_pad(x, min_len):
    """Manually pad tensor to minimum length"""
    if x.shape[-1] >= min_len:
        return x
    pad_len = min_len - x.shape[-1]
    return F.pad(x, (0, pad_len), mode='replicate')

class HybridLayer(nn.Module):
    def __init__(self, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64]):
        super().__init__()
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleList()
        
        # Create all the custom convolution layers
        self._create_increase_filters()
        self._create_decrease_filters() 
        self._create_peak_filters()
        
    def _create_increase_filters(self):
        """Create filters with alternating signs (even indices negative)"""
        for kernel_size in self.kernel_sizes:
            # Create filter with alternating signs
            filter_weight = torch.ones(1, self.input_channels, kernel_size)
            indices = torch.arange(kernel_size)
            filter_weight[0, :, indices % 2 == 0] *= -1
            
            conv = nn.Conv1d(self.input_channels, 1, kernel_size, padding='same', bias=False)
            conv.weight.data = filter_weight
            conv.weight.requires_grad = False
            self.conv_layers.append(conv)
    
    def _create_decrease_filters(self):
        """Create filters with alternating signs (odd indices negative)"""
        for kernel_size in self.kernel_sizes:
            # Create filter with alternating signs
            filter_weight = torch.ones(1, self.input_channels, kernel_size)
            indices = torch.arange(kernel_size)
            filter_weight[0, :, indices % 2 > 0] *= -1
            
            conv = nn.Conv1d(self.input_channels, 1, kernel_size, padding='same', bias=False)
            conv.weight.data = filter_weight
            conv.weight.requires_grad = False
            self.conv_layers.append(conv)
    
    def _create_peak_filters(self):
        """Create peak detection filters with quadratic patterns"""
        for kernel_size in self.kernel_sizes[1:]:  # Skip first kernel size
            extended_size = kernel_size + kernel_size // 2
            filter_weight = torch.zeros(1, self.input_channels, extended_size)
            
            # Create quadratic pattern
            num_points = kernel_size // 4 + 1
            xmash = torch.linspace(0, 1, num_points)[1:].unsqueeze(0).unsqueeze(0)
            
            filter_left = xmash ** 2
            filter_right = torch.flip(filter_left, dims=[2])
            
            # Apply the pattern to different sections
            q = kernel_size // 4
            filter_weight[:, :, 0:q] = -filter_left
            filter_weight[:, :, q:kernel_size//2] = -filter_right
            filter_weight[:, :, kernel_size//2:3*q] = 2 * filter_left
            filter_weight[:, :, 3*q:kernel_size] = 2 * filter_right
            filter_weight[:, :, kernel_size:5*q] = -filter_left
            filter_weight[:, :, 5*q:] = -filter_right
            
            conv = nn.Conv1d(self.input_channels, 1, extended_size, padding='same', bias=False)
            conv.weight.data = filter_weight
            conv.weight.requires_grad = False
            self.conv_layers.append(conv)
    
    def forward(self, x):
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(conv(x))
        
        # Concatenate all outputs
        hybrid_output = torch.cat(conv_outputs, dim=1)
        return F.relu(hybrid_output)

class InceptionModule(nn.Module):
    def __init__(self, input_channels, n_filters, kernel_size, dilation_rate=1, 
                 use_multiplexing=True, use_hybrid_layer=False):
        super().__init__()
        self.use_hybrid_layer = use_hybrid_layer
        self.use_multiplexing = use_multiplexing
        
        if not use_multiplexing:
            n_convs = 1
            n_filters = n_filters * 3
        else:
            n_convs = 3
            
        # Create multiple convolution paths
        self.conv_layers = nn.ModuleList()
        kernel_sizes = [kernel_size // (2**i) for i in range(n_convs)]
        
        for ks in kernel_sizes:
            if ks > 0:  # Ensure kernel size is positive
                self.conv_layers.append(
                    nn.Conv1d(input_channels, n_filters, ks, 
                             padding='same', dilation=dilation_rate, bias=False)
                )
        
        # Hybrid layer if enabled
        if use_hybrid_layer:
            self.hybrid_layer = HybridLayer(input_channels)
        
        # Calculate output channels for batch norm
        total_channels = len(self.conv_layers) * n_filters
        if use_hybrid_layer:
            # Hybrid layer adds: 2 * len(kernel_sizes) + len(kernel_sizes[1:])
            hybrid_channels = 2 * 6 + 5  # Based on default kernel_sizes
            total_channels += hybrid_channels
            
        self.batch_norm = nn.BatchNorm1d(total_channels)
        
    def forward(self, x):
        conv_outputs = []
        
        # Apply all convolution layers
        for conv in self.conv_layers:
            conv_outputs.append(conv(x))
        
        # Apply hybrid layer if enabled
        if self.use_hybrid_layer:
            hybrid_out = self.hybrid_layer(x)
            conv_outputs.append(hybrid_out)
        
        # Concatenate all outputs
        if len(conv_outputs) > 1:
            x = torch.cat(conv_outputs, dim=1)
        else:
            x = conv_outputs[0]
            
        x = self.batch_norm(x)
        x = F.relu(x)
        
        return x

class FCNModule(nn.Module):
    def __init__(self, input_channels, n_filters, kernel_size, dilation_rate=1):
        super().__init__()
        # Using depthwise separable convolution
        self.depthwise = nn.Conv1d(input_channels, input_channels, kernel_size, 
                                  padding='same', dilation=dilation_rate, 
                                  groups=input_channels, bias=False)
        self.pointwise = nn.Conv1d(input_channels, n_filters, 1, bias=False)
        self.batch_norm = nn.BatchNorm1d(n_filters)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class LiteFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32*4,
        padding_mode: str = "replicate",
        kernel_size: int = 41,
        use_custom_filters: bool = True,
        use_dilation: bool = True,
        use_multiplexing: bool = True,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size - 1
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing
        
        # Build the feature extractor
        self.instance_encoder = self._build_encoder()
        
    def _build_encoder(self):
        layers = []
        
        # Initial inception module
        inception = InceptionModule(
            input_channels=self.n_in_channels,
            n_filters=self.out_channels,
            kernel_size=self.kernel_size,
            dilation_rate=1,
            use_multiplexing=self.use_multiplexing,
            use_hybrid_layer=self.use_custom_filters
        )
        layers.append(inception)
        
        # Calculate output channels from inception
        if self.use_multiplexing:
            inception_out_channels = 3 * self.out_channels
        else:
            inception_out_channels = self.out_channels * 3
            
        if self.use_custom_filters:
            # Add hybrid layer channels: 2 * 6 + 5 = 17
            inception_out_channels += 17
        
        # Reduce kernel size
        kernel_size = self.kernel_size // 2
        
        # FCN modules
        current_channels = inception_out_channels
        dilation_rate = 1
        
        for i in range(2):
            if self.use_dilation:
                dilation_rate = 2 ** (i + 1)
                
            fcn = FCNModule(
                input_channels=current_channels,
                n_filters=self.out_channels,
                kernel_size=kernel_size // (2**i) if kernel_size // (2**i) > 0 else 1,
                dilation_rate=dilation_rate
            )
            layers.append(fcn)
            current_channels = self.out_channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle minimum length requirement
        min_len = 41  # Based on max kernel size
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)
