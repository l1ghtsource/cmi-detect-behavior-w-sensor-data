import torch
import torch.nn.functional as F
from torch import nn
from modules.conv_block import manual_pad
from modules.inceptiontime import InceptionBlock

class LetMeCookFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        
        self.instance_encoder = nn.Sequential(
            MultiScaleConv1d(
                n_in_channels, 
                out_channels, 
                kernel_sizes=[3, 5, 7]
            ), # out_channels * 3
            ResidualSEBlock(
                out_channels * 3, 
                out_channels * 4, 
                kernel_size=3, 
                pool_size=1,
                dropout=0.2
            ),  # out_channels * 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)

class XDD_InceptionResnet_FeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ResidualBlock1D(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            InceptionBlock(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)

class Resnet1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ResidualBlock1D(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            ResidualBlock1D(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)
        
class ConvNext1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ConvNeXtBlock1D(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            ConvNeXtBlock1D(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)
        
class EfficientNet1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            MBConv1DBlock(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            MBConv1DBlock(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)
        
class MultiScale1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            MultiScaleBlock1D(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            MultiScaleBlock1D(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)
        
class AFEN1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            AdaptiveExtractionBlock(n_in_channels, out_channels * 4, padding_mode=padding_mode),
            AdaptiveExtractionBlock(out_channels * 4, out_channels * 4, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)

class ResidualBlock1D(nn.Module):
    """1D ResNet block with residual connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        stride: int = 1,
    ):
        super().__init__()
        
        # Main path with two conv layers
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding="same",
            padding_mode=padding_mode,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            padding_mode=padding_mode,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding="same",
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out
    
class ConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt block following the ConvNeXt architecture pattern."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        layer_scale_init_value: float = 1e-6,
        expansion_ratio: int = 4,
    ):
        super().__init__()
        
        # Depthwise convolution (7x7 kernel adapted to 1D)
        self.dwconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            groups=in_channels,
            padding="same",
            padding_mode=padding_mode,
        )
        
        # Layer normalization (ConvNeXt uses LayerNorm instead of BatchNorm)
        self.norm = nn.LayerNorm(in_channels)
        
        # Pointwise/Inverted bottleneck layers
        hidden_channels = expansion_ratio * in_channels
        self.pwconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        
        # Layer scale parameter
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(out_channels),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Layer normalization (need to permute for LayerNorm)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        
        # Pointwise convolutions with expansion
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1) * x
        
        # Residual connection
        x = x + self.shortcut(input_x)
        
        return x
    
class MBConv1DBlock(nn.Module):
    """1D Mobile Inverted Bottleneck block following EfficientNet architecture."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        expansion_ratio: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        
        expanded_channels = in_channels * expansion_ratio
        
        # Expansion phase (pointwise convolution)
        self.expand_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            bias=False,
        )
        self.expand_bn = nn.BatchNorm1d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_channels,
            padding="same",
            padding_mode=padding_mode,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(expanded_channels)
        
        # Squeeze-and-Excitation block
        self.se_block = SE1DBlock(expanded_channels, se_ratio) if se_ratio > 0 else nn.Identity()
        
        # Projection phase (pointwise convolution)
        self.project_conv = nn.Conv1d(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.project_bn = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        
        # Activation
        self.swish = nn.SiLU()  # Swish activation used in EfficientNet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.swish(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.swish(x)
        
        # Squeeze-and-Excitation
        x = self.se_block(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.use_skip_connection:
            x = x + identity
            
        return x

class SE1DBlock(nn.Module):
    """1D Squeeze-and-Excitation block for EfficientNet."""
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        
        reduced_channels = max(1, int(in_channels * se_ratio))
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze
        squeezed = self.squeeze(x)
        
        # Excitation
        excitation = self.excitation(squeezed)
        
        # Scale
        return x * excitation
    
class MultiScaleBlock1D(nn.Module):
    """Multi-Scale block with parallel convolutions of different kernel sizes."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        kernel_sizes: list = [3, 5, 7, 11],
    ):
        super().__init__()
        
        # Calculate output channels per branch
        branch_out_channels = out_channels // len(kernel_sizes)
        
        # Create parallel convolution branches with different kernel sizes
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
            )
            self.branches.append(branch)
        
        # Additional feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through parallel branches
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Concatenate outputs from all branches
        multi_scale_features = torch.cat(branch_outputs, dim=1)
        
        # Fusion layer
        fused_features = self.fusion(multi_scale_features)
        
        # Residual connection
        residual = self.shortcut(x)
        output = fused_features + residual
        
        return output


class MultiScaleAdvancedBlock1D(nn.Module):
    """Advanced Multi-Scale block with dilated convolutions and attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        
        branch_out_channels = out_channels // 4
        
        # Branch 1: Small receptive field (local patterns)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out_channels, kernel_size=3, 
                     padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(branch_out_channels),
            nn.ReLU(),
        )
        
        # Branch 2: Medium receptive field 
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out_channels, kernel_size=7,
                     padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(branch_out_channels),
            nn.ReLU(),
        )
        
        # Branch 3: Large receptive field with dilated convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out_channels, kernel_size=3, dilation=3,
                     padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(branch_out_channels),
            nn.ReLU(),
        )
        
        # Branch 4: Global context with adaptive pooling
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, branch_out_channels, kernel_size=1),
            nn.ReLU(),
        )
        
        # Scale attention mechanism
        self.scale_attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         padding="same", padding_mode=padding_mode),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through different scale branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Global branch (upsample to match sequence length)
        b4 = self.branch4(x)
        b4 = F.interpolate(b4, size=x.shape[-1], mode='linear', align_corners=False)
        
        # Concatenate all branches
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Apply scale attention
        attention_weights = self.scale_attention(multi_scale)
        attended_features = multi_scale * attention_weights
        
        # Residual connection
        residual = self.shortcut(x)
        output = attended_features + residual
        
        return output
    
class AdaptiveExtractionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        
        # Multi-resolution deformable convolutions
        self.deformable_conv = DeformableConv1D(in_channels, out_channels // 4, padding_mode)
        
        # Temporal attention module
        self.temporal_attention = TemporalAttentionModule(in_channels, out_channels // 4)
        
        # Multi-scale wavelet-inspired features
        self.wavelet_features = WaveletFeatureExtractor(in_channels, out_channels // 4, padding_mode)
        
        # Adaptive receptive field module
        self.adaptive_rf = AdaptiveReceptiveFieldModule(in_channels, out_channels // 4, padding_mode)
        
        # Feature fusion with learned weights
        self.feature_fusion = AdaptiveFeatureFusion(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         padding="same", padding_mode=padding_mode),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deformable_feat = self.deformable_conv(x)
        temporal_feat = self.temporal_attention(x)
        wavelet_feat = self.wavelet_features(x)
        adaptive_feat = self.adaptive_rf(x)
        
        combined_features = torch.cat([deformable_feat, temporal_feat, wavelet_feat, adaptive_feat], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        residual = self.shortcut(x)
        return fused_features + residual

class DeformableConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        super().__init__()
        
        self.offset_conv = nn.Conv1d(in_channels, 2 * 9, kernel_size=3, 
                                   padding="same", padding_mode=padding_mode)
        
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, dilation=1,
                     padding="same", padding_mode=padding_mode),
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, dilation=2,
                     padding="same", padding_mode=padding_mode),
            nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, dilation=4,
                     padding="same", padding_mode=padding_mode),
        ])
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(x)
        
        branch_outputs = []
        for conv in self.conv_branches:
            branch_outputs.append(conv(x))
        
        output = torch.cat(branch_outputs, dim=1)
        output = self.norm(output)
        return self.activation(output)

class TemporalAttentionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        query = self.query_conv(x).view(B, -1, L).permute(0, 2, 1)  # B, L, C
        key = self.key_conv(x).view(B, -1, L)  # B, C, L
        value = self.value_conv(x).view(B, -1, L)  # B, C, L
        
        attention = torch.bmm(query, key)  # B, L, L
        attention = self.softmax(attention)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, L
        out = self.gamma * out + x
        
        return out

class WaveletFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        super().__init__()
        
        self.low_pass = nn.Conv1d(in_channels, out_channels // 2, kernel_size=8, 
                                 stride=2, padding=3, padding_mode=padding_mode)
        
        self.high_pass = nn.Conv1d(in_channels, out_channels // 2, kernel_size=8,
                                  stride=2, padding=3, padding_mode=padding_mode)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq = self.low_pass(x)
        high_freq = self.high_pass(x)
        
        low_freq = self.upsample(low_freq)
        high_freq = self.upsample(high_freq)
        
        target_size = x.shape[-1]
        low_freq = low_freq[..., :target_size]
        high_freq = high_freq[..., :target_size]
        
        wavelet_features = torch.cat([low_freq, high_freq], dim=1)
        return self.norm(wavelet_features)

class AdaptiveReceptiveFieldModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        super().__init__()
        
        self.kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     padding="same", padding_mode=padding_mode),
            nn.Conv1d(in_channels, out_channels, kernel_size=5,
                     padding="same", padding_mode=padding_mode),
            nn.Conv1d(in_channels, out_channels, kernel_size=7,
                     padding="same", padding_mode=padding_mode),
            nn.Conv1d(in_channels, out_channels, kernel_size=11,
                     padding="same", padding_mode=padding_mode),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.kernel_predictor(x)  # B, 4, 1
        
        output = 0
        for i, conv in enumerate(self.conv_layers):
            conv_out = conv(x)
            weight = weights[:, i:i+1, :]  # B, 1, 1
            output = output + weight * conv_out
            
        return output

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, total_channels: int):
        super().__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(total_channels, total_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(total_channels // 4, total_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(total_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.channel_attention(x)
        attended_features = x * attention_weights
        
        return self.final_conv(attended_features)
    
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], padding_mode="replicate"):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    ks, 
                    padding="same",
                    padding_mode=padding_mode,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, padding_mode="replicate"):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding="same",
            padding_mode=padding_mode,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            padding="same",
            padding_mode=padding_mode,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = EnhancedSEBlock(out_channels, reduction=8)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    1, 
                    padding="same",
                    padding_mode=padding_mode,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.se(out)
        
        out += shortcut
        out = F.relu(out)
        
        out = self.pool(out)
        out = self.dropout(out)
        
        return out
    
class EnhancedSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)