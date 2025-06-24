"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch.nn.functional as F
from torch import nn
from modules.conv_block import manual_pad

class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            InceptionBlock(n_in_channels, out_channels=out_channels, padding_mode=padding_mode),
            InceptionBlock(out_channels * 4, out_channels, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)


class InceptionBlock(nn.Module):
    """Inception block of three Inception modules, where each module has a residual connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 3,
    ) -> None:
        super().__init__()
        # Create Inception modules that are run sequentially
        inception_modules = []
        for i in range(n_modules):
            inception_modules.append(
                InceptionModule(
                    in_channels=in_channels if i == 0 else out_channels * 4,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                ),
            )
        self.inception_modules = nn.Sequential(*inception_modules)

        # Create residual that is run in parallel to the Inception modules
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=4 * out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=4 * out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_modules = self.inception_modules(x)
        x_residual = self.residual(x)
        return F.relu(x_modules + x_residual)


class InceptionModule(nn.Module):
    """Inception module with bottleneck, conv layers, and max pooling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()

        # Setup bottleneck
        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        # Set up conv layers but don't stack sequentially as these will be run in parallel
        self.conv_layers = nn.ModuleList()
        for kernel_size in [10, 20, 40]:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        # Set up max pooling with bottleneck
        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        self.activation = nn.Sequential(nn.BatchNorm1d(num_features=4 * out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply bottleneck
        x_bottleneck = self.bottleneck(x)
        # Pass through conv layers and max pooling in parallel
        z0 = self.conv_layers[0](x_bottleneck)
        z1 = self.conv_layers[1](x_bottleneck)
        z2 = self.conv_layers[2](x_bottleneck)
        z3 = self.max_pooling_w_bottleneck(x)
        # Stack and pass through activation
        z = torch.cat([z0, z1, z2, z3], dim=1)
        z = self.activation(z)
        return z
    
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * torch.sigmoid(out)

class TemporalAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * torch.sigmoid(out)

class SpectralFeatureExtractor(nn.Module):
    def __init__(self, n_fft: int = 64):
        super().__init__()
        self.n_fft = n_fft
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        original_length = length
        
        if length < self.n_fft:
            x_padded = F.pad(x, (0, self.n_fft - length), mode='replicate')
        else:
            x_padded = x
        
        fft = torch.fft.fft(x_padded, n=self.n_fft, dim=-1)
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        psd = magnitude ** 2
        
        spectral_features = torch.cat([magnitude, phase, psd], dim=1)
        
        if spectral_features.shape[-1] != original_length:
            spectral_features = F.interpolate(
                spectral_features, 
                size=original_length, 
                mode='linear', 
                align_corners=False
            )
        
        return spectral_features

class EnhancedInceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        kernel_sizes = [5, 11, 23, 41]
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        self.dilated_convs = nn.ModuleList()
        dilations = [2, 4, 8]
        for dilation in dilations:
            self.dilated_convs.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                    padding_mode=padding_mode,
                )
            )

        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        total_out_channels = 8 * out_channels
        
        self.activation = nn.Sequential(
            nn.BatchNorm1d(num_features=total_out_channels), 
            nn.ReLU()
        )
        
        self.channel_attention = ChannelAttention(total_out_channels)
        self.temporal_attention = TemporalAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_outputs.append(conv_layer(x_bottleneck))
        
        dilated_outputs = []
        for dilated_conv in self.dilated_convs:
            dilated_outputs.append(dilated_conv(x_bottleneck))
        
        max_pool_output = self.max_pooling_w_bottleneck(x)
        
        z = torch.cat(conv_outputs + dilated_outputs + [max_pool_output], dim=1)
        
        z = self.activation(z)
        
        z = self.channel_attention(z)
        z = self.temporal_attention(z)
        
        return z

class FeaturePyramidInception(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = 32,
        padding_mode: str = "replicate"
    ):
        super().__init__()
        
        self.level1 = EnhancedInceptionModule(in_channels, out_channels, padding_mode=padding_mode)  # Полное разрешение
        self.level2 = EnhancedInceptionModule(in_channels, out_channels, padding_mode=padding_mode)  # 1/2 разрешение
        self.level3 = EnhancedInceptionModule(in_channels, out_channels, padding_mode=padding_mode)  # 1/4 разрешение
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        
        self.total_channels = 3 * 8 * out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.level1(x)
        
        x_down2 = F.avg_pool1d(x, 2)
        x2 = self.level2(x_down2)
        
        x_down4 = F.avg_pool1d(x, 4)
        x3 = self.level3(x_down4)
        
        original_size = x1.shape[-1]
        x2_up = self.upsample2(x2)
        x3_up = self.upsample4(x3)
        
        if x2_up.shape[-1] != original_size:
            x2_up = F.interpolate(x2_up, size=original_size, mode='linear', align_corners=False)
        if x3_up.shape[-1] != original_size:
            x3_up = F.interpolate(x3_up, size=original_size, mode='linear', align_corners=False)
        
        return torch.cat([x1, x2_up, x3_up], dim=1)

class EnhancedInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 2,
    ) -> None:
        super().__init__()
        
        inception_modules = []
        for i in range(n_modules):
            inception_modules.append(
                FeaturePyramidInception(
                    in_channels=in_channels if i == 0 else self._get_output_channels(out_channels),
                    out_channels=out_channels,
                    padding_mode=padding_mode,
                )
            )
        self.inception_modules = nn.Sequential(*inception_modules)

        final_out_channels = self._get_output_channels(out_channels)
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=final_out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=final_out_channels),
        )

    def _get_output_channels(self, out_channels: int) -> int:
        return 3 * 8 * out_channels  # 3 levels * 8 * out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_modules = self.inception_modules(x)
        x_residual = self.residual(x)
        return F.relu(x_modules + x_residual)

class EnhancedInceptionTimeFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
        use_spectral_features: bool = True,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.use_spectral_features = use_spectral_features
        
        if use_spectral_features:
            self.spectral_extractor = SpectralFeatureExtractor(n_fft=64)
            total_in_channels = n_in_channels + n_in_channels * 3  # original + 3 * original
        else:
            total_in_channels = n_in_channels
        
        self.instance_encoder = nn.Sequential(
            EnhancedInceptionBlock(
                total_in_channels, 
                out_channels=out_channels, 
                padding_mode=padding_mode
            ),
            EnhancedInceptionBlock(
                3 * 8 * out_channels,
                out_channels, 
                padding_mode=padding_mode
            ),
        )
        
        self.final_projection = nn.Sequential(
            nn.Conv1d(3 * 8 * out_channels, out_channels * 4, 1),
            nn.BatchNorm1d(out_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 64
        
        if x.shape[-1] >= min_len:
            processed_x = x
        else:
            processed_x = manual_pad(x, min_len)
        
        if self.use_spectral_features:
            spectral_features = self.spectral_extractor(processed_x)
            processed_x = torch.cat([processed_x, spectral_features], dim=1)
        
        features = self.instance_encoder(processed_x)
        
        output = self.final_projection(features)
        
        return output