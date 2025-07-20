import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.medformer_encdec import Encoder, EncoderLayer
from modules.selfattn import MedformerLayer
from modules.embed import ListPatchEmbedding
from configs.config import cfg

# bad solo, bad in hybrid :(

class Original_Medformer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2405.19363
    """
    def __init__(self, configs):
        super(Original_Medformer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel
        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        # augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.seq_len,
            patch_len_list,
            stride_list,
            configs.dropout,
            # augmentations,
            configs.single_channel,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model
                * len(patch_num_list)
                * (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

# for imu only
class Medformer_SingleSensor_v1(nn.Module):
    def __init__(
        self,
        seq_len=cfg.seq_len,
        n_vars=cfg.imu_vars,
        num_classes=cfg.main_num_classes,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        patch_len_list="2,4,8,16,32",
        activation="gelu",
        output_attention=False,
        no_inter_attn=False,
        single_channel=False
    ):
        super(Medformer_SingleSensor_v1, self).__init__()
        
        # Core parameters
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.d_model = d_model
        self.single_channel = single_channel
        self.output_attention = output_attention
        
        # Patch embedding parameters
        patch_len_list = list(map(int, patch_len_list.split(",")))
        stride_list = patch_len_list
        
        # Calculate patch numbers for each patch length
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        self.patch_num_list = patch_num_list
        
        # Patch Embedding Layer
        self.enc_embedding = ListPatchEmbedding(
            n_vars,
            d_model,
            seq_len,
            patch_len_list,
            stride_list,
            dropout,
            single_channel,
        )
        
        # Encoder with Medformer layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        d_model,
                        n_heads,
                        dropout,
                        output_attention,
                        no_inter_attn,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        # Classification head
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        
        # Calculate projection input dimension
        projection_dim = (
            d_model * len(patch_num_list) * 
            (1 if not single_channel else n_vars)
        )
        
        self.projection1 = nn.Linear(projection_dim, num_classes)
        self.projection2 = nn.Linear(projection_dim, 2)

    def forward(self, x, pad_mask=None):
        # Input shape: (batch, 1, seq_len, n_vars)
        # Need to reshape to (batch, seq_len, n_vars) for embedding
        batch_size = x.shape[0]
        x_enc = x.squeeze(1)  # Remove the channel dimension: (batch, seq_len, n_vars)
        
        # Patch embedding
        enc_out = self.enc_embedding(x_enc)
        
        # Encoder forward pass
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Handle single channel case
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.n_vars, *enc_out.shape[-2:]))

        # Apply activation and dropout
        output = self.activation(enc_out)
        output = self.dropout(output)
        
        # Flatten for classification
        output = output.reshape(output.shape[0], -1)
        
        # Final classification projection
        output1 = self.projection1(output)
        output2 = self.projection2(output)
        
        return output1, output2
    
class MultiSensor_Medformer_v1(nn.Module):
    """
    IMU (1 × 7 vars) + ToF (5 × 64 vars) + Thermal (5 × 1 vars)
    — multiscale Medformer with separate intra-sensor encoders and a
    cross-sensor encoder that works on the *list-of-scales* representation
    expected by MedformerLayer.
    """
    def __init__(
        self,
        seq_len=cfg.seq_len,
        imu_vars=cfg.imu_vars,
        tof_vars=64,
        thm_vars=1,
        num_imu_sensors=1,
        num_tof_sensors=5,
        num_thm_sensors=5,
        num_classes=cfg.main_num_classes,
        d_model=128,
        n_heads=8,
        intra_layers=2,
        cross_layers=2,
        d_ff=256,
        dropout=0.1,
        patch_len_list="2,4,8,8,16,16,16,32,32,32,32,32",
        activation="gelu",
        output_attention=False,
        no_inter_attn=False,
    ):
        super().__init__()

        # ─────────── basic parameters ───────────
        self.seq_len = seq_len
        self.patch_len_list = list(map(int, patch_len_list.split(",")))
        self.patch_num_list = [
            (seq_len - p) // p + 2 for p in self.patch_len_list
        ]  # Li for every scale
        self.num_scales = len(self.patch_len_list)

        # projection dim for the classifier
        total_tokens = sum(self.patch_num_list) * (
            num_imu_sensors + num_tof_sensors + num_thm_sensors
        )
        self.projection_dim = d_model * total_tokens

        # ─────────── patch embeddings ───────────
        stride_list = self.patch_len_list
        self.imu_embedding = ListPatchEmbedding(
            imu_vars, d_model, seq_len, self.patch_len_list, stride_list, dropout, False
        )
        self.tof_embedding = ListPatchEmbedding(
            tof_vars, d_model, seq_len, self.patch_len_list, stride_list, dropout, False
        )
        self.thm_embedding = ListPatchEmbedding(
            thm_vars, d_model, seq_len, self.patch_len_list, stride_list, dropout, False
        )

        # ─────────── encoders ───────────
        self.imu_encoder = self._build_encoder(
            d_model, n_heads, intra_layers, d_ff, dropout, activation, output_attention, no_inter_attn, patch_len_list=self.patch_len_list
        )
        self.tof_encoder = self._build_encoder(
            d_model, n_heads, intra_layers, d_ff, dropout, activation, output_attention, no_inter_attn, patch_len_list=self.patch_len_list
        )
        self.thm_encoder = self._build_encoder(
            d_model, n_heads, intra_layers, d_ff, dropout, activation, output_attention, no_inter_attn, patch_len_list=self.patch_len_list
        )
        self.cross_encoder = self._build_encoder(
            d_model, n_heads, cross_layers, d_ff, dropout, activation, output_attention, no_inter_attn, patch_len_list=self.patch_len_list
        )

        # ─────────── sensor-type tokens ───────────
        self.register_parameter(
            "sensor_type_embedding", nn.Parameter(torch.randn(3, d_model))
        )

        # ─────────── classification head ───────────
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection1 = nn.Linear(self.projection_dim, num_classes)
        self.projection2 = nn.Linear(self.projection_dim, 2)
        self.projection3 = nn.Linear(self.projection_dim, 4)

        # counts needed later
        self.num_imu_sensors = num_imu_sensors
        self.num_tof_sensors = num_tof_sensors
        self.num_thm_sensors = num_thm_sensors

    # ──────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_encoder(
        d_model, n_heads, layers, d_ff, dropout, activation, output_attention, no_inter_attn, patch_len_list
    ):
        return Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        num_scales=len(patch_len_list.split(",")),
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        output_attention=output_attention,
                        no_inter_attn=no_inter_attn,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def _process_sensor_group(self, data, embedding, encoder, sensor_type_idx):
        """
        data shape: [B, N_sensors, L, vars]
        Returns list of length `num_scales`,
        each tensor [B, (N_sensors * Li), d_model]
        """
        B, N, _, _ = data.shape
        scale_buffers = [[] for _ in range(self.num_scales)]  # gather per scale

        sensor_token = self.sensor_type_embedding[sensor_type_idx].view(1, 1, -1)

        for s_idx in range(N):
            x = data[:, s_idx]  # [B, L, vars]
            scale_list = embedding(x)  # list[Tensor] len = num_scales

            # add sensor-type embedding & intra-sensor transformer
            scale_list = [t + sensor_token for t in scale_list]
            encoded_list, _ = encoder(scale_list, attn_mask=None)

            for sc in range(self.num_scales):
                scale_buffers[sc].append(encoded_list[sc])

        # concat sensors along token dimension for every scale
        return [torch.cat(buf, dim=1) for buf in scale_buffers]

    # ──────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        # -------- per-type encoding --------
        imu_feats = self._process_sensor_group(imu_data, self.imu_embedding,
                                               self.imu_encoder, sensor_type_idx=0)
        tof_feats = self._process_sensor_group(tof_data, self.tof_embedding,
                                               self.tof_encoder, sensor_type_idx=1)
        thm_feats = self._process_sensor_group(thm_data, self.thm_embedding,
                                               self.thm_encoder, sensor_type_idx=2)

        # -------- merge three sensor types (scale by scale) --------
        cross_input = [
            torch.cat([imu_feats[s], tof_feats[s], thm_feats[s]], dim=1)
            for s in range(self.num_scales)
        ]  # list[len=num_scales]

        # -------- cross-sensor encoder --------
        cross_out, _ = self.cross_encoder(cross_input, attn_mask=None)

        # -------- classification head --------
        tokens = torch.cat(cross_out, dim=1)          # [B, total_tokens, d_model]
        x = self.dropout(self.activation(tokens))
        x = x.flatten(start_dim=1)                    # [B, projection_dim]

        out1 = self.projection1(x)
        out2 = self.projection2(x)
        out3 = self.projection3(x)
        return out1, out2, out3