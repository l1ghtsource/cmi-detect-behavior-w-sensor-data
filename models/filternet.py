# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from configs.config import cfg

# good in hybrid, solo ok

DEFAULT_WIDTH = 100

class BaseNet(nn.Module):
    """ Abstract 'base' network that can be reimplemented for specific architectures."""

    def __init__(
        self,
        input_channels=cfg.imu_vars,
        num_output_classes=[cfg.main_num_classes, 2],
        output_type="many_to_one_takelast",
        keep_intermediates=False,
        do_multi=False,
        **other_kwargs,
    ):
        self.do_multi = do_multi

        self.output_type = output_type
        self.num_output_classes = num_output_classes
        self.input_channels = input_channels
        self.keep_intermediates = keep_intermediates
        self.padding_lost_per_side = 0
        self.output_stride = 1

        super(BaseNet, self).__init__()

        self.build(**other_kwargs)

    def build(self, **other_kwargs):
        """ Builds the network. Can take any number of custom params as kwargs to configure it.
        REIMPLEMENT IN SUBCLASSES.
        """
        raise NotImplementedError()

    def forward(self, X, pad_mask=None):
        ys = self._forward(X, pad_mask)

        if self.output_type == "many_to_one_takelast":
            return [y[:, :, -1] for y in ys]
        elif self.output_type == "many_to_many":
            return ys
        else:
            raise NotImplemented(self.output_type)

    def _forward(self, X, pad_mask=None):
        """Forward pass logic specific to this network type.
        REIPMLEMENT IN SUBCLASSES.
        Input dimensionality: (N, C_{in}, L_{in})"""
        raise NotImplementedError()

    def transform_targets(self, ys, one_hot=True):
        """ Convert a `y` vector (one of `ys`) into targets that can be compared
        to network outputs... take into account padding, one-hot encoding (if requested),
        and whether the network is many-to-many or many-to-one. """
        ys2 = []
        for i_y, y in enumerate(ys):
            if self.output_type == "many_to_one_takelast" and not one_hot:
                ys2.append(y[:, [-1]])
                continue

            # Account for any change in sequence length due to padding
            if self.padding_lost_per_side > 0:
                y = y[:, self.padding_lost_per_side : -self.padding_lost_per_side]

            # for many-to-many, if needed:
            win_len = y.shape[-1]
            # Calculate number of outputs. This is not always accurate and sometimes
            # 'floor' needs to change to 'ceil' or vice-versa... TBD is to implement
            # a system that calculates this accurately for all of our possible
            # architectures.
            output_size = int(np.floor(win_len / float(self.output_stride)))
            # Now, create that many outputs, evenly spaced by output_stride
            output_idxs = np.arange(output_size) * self.output_stride
            # Now, center it in the middle of the window. Depending on our
            #  architecture, this many not be *exactly* optimal, but it's
            #  a good guess on average.
            # Note: win_len - 1 because of zero-indexing
            output_idxs = np.round(
                output_idxs - (output_idxs.mean() - (win_len - 1) / 2.0)
            ).astype(int)

            if one_hot:
                if len(y.shape) == 2:
                    # Do one-hot encoding
                    y = torch.zeros(
                        y.size()[0],
                        self.num_output_classes[i_y],
                        y.size()[1],
                        device=y.device,
                    ).scatter_(1, y.unsqueeze(1), 1)

                if self.output_type == "many_to_one_takelast":
                    ys2.append(y[:, :, [output_idxs[-1]]])
                elif self.output_type == "many_to_many":
                    ys2.append(y[:, :, output_idxs])
                else:
                    raise NotImplemented(self.output_type)

            else:
                if self.output_type == "many_to_one_takelast":
                    ys2.append(y[:, [output_idxs[-1]]])
                elif self.output_type == "many_to_many":
                    ys2.append(y[:, output_idxs])
                else:
                    raise NotImplemented(self.output_type)

        return ys2
    
class CustomRNNMixin(object):
    """Mixin to hekp wraps PyTorch recurrent layer(s) to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    """

    def __init__(self, *args, **kwargs):
        if "batch_first" not in kwargs:
            kwargs["batch_first"] = True
        super().__init__(*args, **kwargs)

    def forward(self, input):
        input = input.transpose(1, 2).contiguous()
        output, h_n = super().forward(input)
        return output.transpose(1, 2).contiguous()

class CustomGRU(CustomRNNMixin, nn.GRU):
    """Wraps PyTorch GRU to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    GRU sub-type.
    """

    pass

class CustomLSTM(CustomRNNMixin, nn.LSTM):
    """Wraps PyTorch LSTM version to swap axes 1&2 (and back) since that's what PyTorch RNNs expect.
    LSTM sub-type.
    """

    pass

class CGLLayer(nn.Sequential):
    """Flexible mplementation of a convolution/GRU/LSTM layer, which is the basic building block of our models. Each
    layer is made up of (optional) dropout, a CNN, GRU, or LSTM layer surrounded by (optional) striding/pooling
    layers, and a BatchNorm layer.

    This layer subclasses torch.nn.Sequential so that all the pytorch magic still works with it (like transferring
    to/from devices, initializing weights, switching back/forth to eval mode, etc)
    """

    output_size = (
        None
    )  # type: int # depth (channels) output by this layer, useful for hooking up to subsequent modules.

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=5,
        type="cnn",
        stride=1,
        pool=None,
        dropout=0.1,
        stride_pos=None,
        batch_norm=True,
        groups=1,
    ):
        """

        Parameters
        ----------
        input_size: int
            Depth (channels) of input / previous layer
        output_size: int
            Depth (channels) that this layer will output
        kernel_size: int
            For CNNs
        type: str
            'cnn', 'lstm', or 'gru'; determines primary layer type.
        stride: int
            How much to decimate output (in temporal dimension) via _striding_. Defaults to 1 (no decimation).
        pool: int
            How much to decimate output (in temporal dimension) via _average_pooling_. Defaults to 1 (no decimation).
        dropout: float
            Amount of dropout Defaults to 0.0, i.e., None
        stride_pos: str
            For recurrent layers only, determines whether striding/pooling is done *before* (default) or
            *after* the recurrent layer.
        batch_norm: bool
            If True (default), the activation layer is followed by a batchnorm layer.
        """

        layers = []
        self.output_size = output_size

        if type == "cnn":
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            s = 1 if pool else stride
            p = int(np.ceil((kernel_size - s) / 2.0))
            layers.append(
                nn.Conv1d(
                    input_size,
                    output_size,
                    stride=s,
                    kernel_size=kernel_size,
                    padding=p,
                    groups=groups,
                )
            )
            layers.append(nn.ReLU())
            if pool:
                p = int(np.ceil((pool - stride) / 2.0))
                layers.append(
                    nn.AvgPool1d(pool, stride, padding=p, count_include_pad=False)
                )
        elif type in ["gru", "lstm"]:
            klass = {"gru": CustomGRU, "lstm": CustomLSTM}[type]
            if (pool or stride) and stride_pos != "post":
                pl = 1 if not pool else pool
                p = np.ceil((pl - stride) / 2.0).astype(int)
                layers.append(nn.AvgPool1d(pl, stride=stride, padding=p))
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            assert output_size % 2 == 0  # must be even b/c bidirectional
            layers.append(
                klass(
                    input_size=input_size,
                    hidden_size=int(output_size / 2),
                    bidirectional=True,
                )
            )
            if (pool or stride) and stride_pos == "post":
                pl = 1 if not pool else pool
                p = np.ceil((pl - stride) / 2.0).astype(int)
                layers.append(nn.AvgPool1d(pl, stride=stride, padding=p))
        else:
            raise ValueError("Unknown layer type: %s" % type)

        # Follow with BN
        if batch_norm:
            layers.append(nn.BatchNorm1d(self.output_size))

        super().__init__(*layers)

class FilterNet_SingleSensor_Test(BaseNet):
    def build(
        self,
        n_pre=1,
        w_pre=DEFAULT_WIDTH,
        n_strided=3,
        w_strided=DEFAULT_WIDTH,
        n_interp=4,
        w_interp=DEFAULT_WIDTH,
        n_dense_pre_l=1,
        w_dense_pre_l=DEFAULT_WIDTH,
        n_l=1,
        w_l=DEFAULT_WIDTH,
        n_dense_post_l=0,
        w_dense_post_l=int(DEFAULT_WIDTH / 2),
        cnn_kernel_size=5,
        scale=1.0,
        bn_pre=False,
        dropout=0.1,
        do_pool=True,
        stride_pos="post",
        stride_amt=2,
        **other_kwargs,
    ):
        if self.do_multi:
            w_pre = 2 * w_pre
            w_strided = 2 * w_strided
            w_interp = 2 * w_interp
            w_dense_pre_l = 2 * w_dense_pre_l
            w_l = 2 * w_l
            w_dense_post_l = 2 * w_dense_post_l
            
        # if scale != 1:
        w_pre = int((w_pre * scale))  # / 6) * 6
        w_strided = int((w_strided * scale))  # / 6) * 6
        w_interp = int(w_interp * scale)
        w_dense_pre_l = int(w_dense_pre_l * scale)
        w_l = int((w_l * scale) / 2) * 2
        w_dense_post_l = int(w_dense_post_l * scale)

        down_stack_1 = []
        in_shape = self.input_channels

        if bn_pre:
            down_stack_1.append(nn.BatchNorm1d(in_shape))

        for i in range(n_pre):
            down_stack_1.append(
                CGLLayer(in_shape, w_pre, cnn_kernel_size, type="cnn", dropout=dropout)
            )
            in_shape = down_stack_1[-1].output_size

        for i in range(n_strided):
            stride = stride_amt
            pool = stride if (do_pool and stride > 1) else None
            ltype = "cnn"
            down_stack_1.append(
                CGLLayer(
                    in_shape,
                    w_strided,
                    cnn_kernel_size,
                    type=ltype,
                    stride=stride,
                    pool=pool,
                    stride_pos=stride_pos,
                    dropout=dropout,
                    # groups=3 if ( i % 2 == 0 ) else 2
                )
            )
            self.output_stride *= stride
            in_shape = down_stack_1[-1].output_size
        ds_1_end_size = in_shape
        self.down_stack_1 = nn.Sequential(*down_stack_1)

        ds2_ltype = "cnn"
        down_stack_2 = []

        for i in range(n_interp):
            stride = stride_amt if (i < n_interp - 1) else 1
            pool = stride if (do_pool and stride > 1) else None
            w = int(np.ceil(w_interp * 0.5 ** (i + 1)))
            # if i == n_interp-1:
            #     w = int(w_interp * .66)
            # if i == n_interp - 2:
            #     w =int(w_interp * .33)
            # else:
            #     w = w_interp
            down_stack_2.append(
                CGLLayer(
                    in_shape,
                    w,
                    cnn_kernel_size,
                    type=ds2_ltype,
                    stride=stride,
                    pool=pool,
                    stride_pos=stride_pos,
                    dropout=dropout,
                    # groups = 3 if ( i % 2 == 0 ) else 2
                )
            )
            in_shape = down_stack_2[-1].output_size

        self.down_stack_2 = nn.Sequential(*down_stack_2)

        self.merged_output_size = ds_1_end_size + sum(
            [l.output_size for l in down_stack_2]
        )

        in_shape = self.merged_output_size

        lstm_stack = []
        for i in range(n_dense_pre_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape, w_dense_pre_l, kernel_size=1, type="cnn", dropout=dropout
                )
            )
            in_shape = lstm_stack[-1].output_size

        for i in range(n_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape,
                    w_l,
                    cnn_kernel_size,  # unused when type!-='cnn'
                    type="lstm",
                    dropout=dropout,
                )
            )
            in_shape = lstm_stack[-1].output_size

        for i in range(n_dense_post_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape, w_dense_post_l, kernel_size=1, type="cnn", dropout=dropout
                )
            )
            in_shape = lstm_stack[-1].output_size

        self.lstm_stack = nn.Sequential(*lstm_stack)

        # [batch, chan, seq]

        end_stacks = []
        for num_output_classes in self.num_output_classes:
            end_stacks.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    #     # This sort of Conv1D acts as a time-distributed Dense layer.
                    nn.Linear(in_shape, num_output_classes),
                    # nn.Conv1d(
                    #     in_shape, num_output_classes, 1
                    # ),  # time-distributed dense
                )
                # CGLLayer(
                #     in_shape,
                #     num_output_classes,
                #     kernel_size=1,
                #     type="cnn",
                #     dropout=dropout,
                #     batch_norm=False
                # )
            )

        self.end_stacks = nn.ModuleList(end_stacks)

    def _forward(self, X, pad_mask=None):
        """(N, C_{in}, L_{in})"""
        X = X[:, 0, :, :].transpose(1, 2) # [batch, chan, seq]
        Xs = [X]  
        Xs.append(self.down_stack_1(Xs[-1]))

        to_merge = [Xs[-1]]
        for module in self.down_stack_2:
            output = module(Xs[-1])
            Xs.append(output)
            to_merge.append(
                F.interpolate(
                    output,
                    size=to_merge[0].shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            )

        merged = torch.cat(to_merge, dim=1)
        Xs.append(merged)
        Xs.append(self.lstm_stack(Xs[-1]))

        if self.keep_intermediates:
            self.Xs = Xs

        ys = []

        # (N, C_{in}, L_{in})

        for end_stack in self.end_stacks:
            # (N, C_{in}, L_{in}) => # (N, L_{in},  C_{in},)
            x = Xs[-1].permute([0, 2, 1])
            x = end_stack(x)
            x = x.permute([0, 2, 1])
            ys.append(x)

        # ys = [end_stack(Xs[-1]) for end_stack in self.end_stacks]

        # No softmax because the pytorch cross_entropy loss function wants the raw outputs.

        return ys
    
class FilterNetFeatureExtractor(FilterNet_SingleSensor_Test):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.end_stacks = nn.ModuleList()
    
    def _forward(self, X, pad_mask=None):
        X = X[:, 0, :, :].transpose(1, 2)          # [B, C, T]
        Xs = [X]
        Xs.append(self.down_stack_1(Xs[-1]))

        to_merge = [Xs[-1]]
        for module in self.down_stack_2:
            out = module(Xs[-1])
            Xs.append(out)
            to_merge.append(
                F.interpolate(out, size=to_merge[0].shape[-1],
                              mode="linear", align_corners=False)
            )

        merged = torch.cat(to_merge, dim=1)
        Xs.append(merged)
        Xs.append(self.lstm_stack(Xs[-1]))         # [B, hidden_dim, T']

        if self.keep_intermediates:
            self.Xs = Xs

        feats = Xs[-1]                             # [B, hidden_dim, T']
        feats = feats[:, :, -1]                    # [B, hidden_dim]
        return feats

    def forward(self, X, pad_mask=None):
        return self._forward(X, pad_mask)
    
class FilterNet_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 head_droupout=0.2,
                 attention_n_heads=10,
                 attention_dropout=0.2,
                 num_classes=cfg.main_num_classes):
        super().__init__()
        
        self.channel_sizes = {
            'imu': 3,      # x_imu: 0-2
            'rot': 4,      # x_rot: 3-6  
            'fe1': 13+3,     # x_fe1: 7-19+3
            'fe2': 9,      # x_fe2: 20+3-28+3
            'full': 29+3     # x_full: 0-28+3
        }
        
        self.branch_extractors = nn.ModuleDict()
        
        for branch_name, channel_size in self.channel_sizes.items():
            self.branch_extractors[f'{branch_name}_extractor1'] = FilterNetFeatureExtractor(
                input_channels=channel_size,
                do_multi=False
            )

        final_hidden_dim = (DEFAULT_WIDTH) * 5
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=DEFAULT_WIDTH,
            num_heads=attention_n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(DEFAULT_WIDTH)

        final_feature_dim = final_hidden_dim * 1

        self.head1 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, num_classes)
        )

        self.head2 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 2)
        )

        self.head3 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 4)
        )
            
    def process_extractor(self, x_dict, pad_mask=None):
        branch_features = []
        
        for branch_name in self.channel_sizes.keys():
            x = x_dict[branch_name]
            feature = self.branch_extractors[f'{branch_name}_extractor1'](x)
            branch_features.append(feature)
        
        stacked_features = torch.stack(branch_features, dim=1)
        attended_features, _ = self.self_attention(
            stacked_features, 
            stacked_features, 
            stacked_features
        )
        attended_features = self.attention_norm(attended_features + stacked_features)
        final_features = attended_features.view(attended_features.size(0), -1)
        
        return final_features
    
    def forward(self, _x, pad_mask=None):
        # input is (bs, 1, T, C)
        
        x_dict = {
            'imu': _x[:, :, :, :3],
            'rot': _x[:, :, :, 3:7],
            'fe1': _x[:, :, :, 7:20+3],
            'fe2': _x[:, :, :, 20+3:29+3],
            'full': _x
        }
        
        final_features = self.process_extractor(x_dict, pad_mask=pad_mask)

        out1 = self.head1(final_features)
        out2 = self.head2(final_features)
        out3 = self.head3(final_features)

        return out1, out2, out3