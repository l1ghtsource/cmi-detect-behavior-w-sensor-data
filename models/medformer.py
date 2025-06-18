import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.medformer_encdec import Encoder, EncoderLayer
from modules.selfattn import MedformerLayer
from modules.embed import ListPatchEmbedding
from configs.config import cfg

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
        n_vars=7,
        num_classes=cfg.main_num_classes,
        d_model=128,
        n_heads=8,
        e_layers=6,
        d_ff=256,
        dropout=0.1,
        patch_len_list="2,4,8,8,16,16,16,32,32,32,32,32",
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
    def __init__(
        self,
        seq_len=cfg.seq_len,
        imu_vars=7,
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
        no_inter_attn=False
    ):
        super(MultiSensor_Medformer_v1, self).__init__()
        
        # Core parameters
        self.seq_len = seq_len
        self.imu_vars = imu_vars
        self.tof_vars = tof_vars
        self.thm_vars = thm_vars
        self.num_imu_sensors = num_imu_sensors
        self.num_tof_sensors = num_tof_sensors
        self.num_thm_sensors = num_thm_sensors
        self.num_classes = num_classes
        self.d_model = d_model
        self.output_attention = output_attention
        
        # Patch parameters
        patch_len_list = list(map(int, patch_len_list.split(",")))
        stride_list = patch_len_list
        
        # Calculate patch numbers
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        self.patch_num_list = patch_num_list
        
        # === INTRA-SENSOR PROCESSING ===
        
        # IMU Patch Embedding (for each IMU sensor)
        self.imu_embedding = ListPatchEmbedding(
            imu_vars, d_model, seq_len, patch_len_list, stride_list, dropout, False
        )
        
        # ToF Patch Embedding (for each ToF sensor)
        self.tof_embedding = ListPatchEmbedding(
            tof_vars, d_model, seq_len, patch_len_list, stride_list, dropout, False
        )
        
        # Thermal Patch Embedding (for each thermal sensor)
        self.thm_embedding = ListPatchEmbedding(
            thm_vars, d_model, seq_len, patch_len_list, stride_list, dropout, False
        )
        
        # Intra-sensor encoders
        self.imu_encoder = self._build_encoder(
            len(patch_len_list), d_model, n_heads, intra_layers, d_ff, 
            dropout, activation, output_attention, no_inter_attn
        )
        
        self.tof_encoder = self._build_encoder(
            len(patch_len_list), d_model, n_heads, intra_layers, d_ff,
            dropout, activation, output_attention, no_inter_attn
        )
        
        self.thm_encoder = self._build_encoder(
            len(patch_len_list), d_model, n_heads, intra_layers, d_ff,
            dropout, activation, output_attention, no_inter_attn
        )
        
        # === CROSS-SENSOR PROCESSING ===
        
        # Cross-sensor encoder
        self.cross_encoder = self._build_encoder(
            len(patch_len_list), d_model, n_heads, cross_layers, d_ff,
            dropout, activation, output_attention, no_inter_attn
        )
        
        # Sensor type embeddings
        self.sensor_type_embedding = nn.Parameter(
            torch.randn(3, d_model)  # 3 sensor types: IMU, ToF, Thermal
        )
        
        # === CLASSIFICATION HEAD ===
        
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total projection dimension
        total_sensors = num_imu_sensors + num_tof_sensors + num_thm_sensors
        projection_dim = d_model * len(patch_num_list) * total_sensors
        
        self.projection1 = nn.Linear(projection_dim, num_classes)
        self.projection2 = nn.Linear(projection_dim, 2)
    
    def _build_encoder(self, num_patches, d_model, n_heads, layers, d_ff, 
                      dropout, activation, output_attention, no_inter_attn):
        """Build encoder with specified parameters"""
        return Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        num_patches, d_model, n_heads, dropout,
                        output_attention, no_inter_attn
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation
                )
                for _ in range(layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
    
    def _process_sensor_data(self, data, embedding, encoder, sensor_type_idx):
        """
        Process data from one sensor type
        
        Args:
            data: [B, num_sensors, L, vars]
            embedding: Patch embedding layer
            encoder: Intra-sensor encoder
            sensor_type_idx: Index for sensor type embedding
            
        Returns:
            Processed sensor features [B, num_sensors, total_patches, d_model]
        """
        B, num_sensors, L, vars = data.shape
        
        # Process each sensor separately
        sensor_outputs = []
        
        for i in range(num_sensors):
            # Get data for one sensor: [B, L, vars]
            sensor_data = data[:, i, :, :]
            
            # Patch embedding
            embedded = embedding(sensor_data)  # [B, total_patches, d_model]
            
            # Add sensor type embedding
            embedded = embedded + self.sensor_type_embedding[sensor_type_idx].unsqueeze(0).unsqueeze(0)
            
            # Intra-sensor encoding
            encoded, _ = encoder(embedded, attn_mask=None)  # [B, total_patches, d_model]
            
            sensor_outputs.append(encoded)
        
        # Stack outputs: [B, num_sensors, total_patches, d_model]
        return torch.stack(sensor_outputs, dim=1)
    
    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        """
        B = imu_data.shape[0]
        
        # === INTRA-SENSOR PROCESSING ===
        
        # Process each sensor type
        imu_features = self._process_sensor_data(
            imu_data, self.imu_embedding, self.imu_encoder, sensor_type_idx=0
        )  # [B, 1, total_patches, d_model]
        
        tof_features = self._process_sensor_data(
            tof_data, self.tof_embedding, self.tof_encoder, sensor_type_idx=1  
        )  # [B, 5, total_patches, d_model]
        
        thm_features = self._process_sensor_data(
            thm_data, self.thm_embedding, self.thm_encoder, sensor_type_idx=2
        )  # [B, 5, total_patches, d_model]
        
        # === CROSS-SENSOR PROCESSING ===
        
        # Concatenate all sensor features along sensor dimension
        # [B, (1+5+5), total_patches, d_model] = [B, 11, total_patches, d_model]
        all_features = torch.cat([imu_features, tof_features, thm_features], dim=1)
        
        # Reshape for cross-sensor attention: [B, 11*total_patches, d_model]
        B, total_sensors, total_patches, d_model = all_features.shape
        cross_input = all_features.reshape(B, total_sensors * total_patches, d_model)
        
        # Cross-sensor encoding
        cross_output, attns = self.cross_encoder(cross_input, attn_mask=None)
        
        # === CLASSIFICATION ===
        
        # Apply activation and dropout
        output = self.activation(cross_output)
        output = self.dropout(output)
        
        # Flatten for classification: [B, total_sensors * total_patches * d_model]
        output = output.reshape(B, -1)
        
        # Final projection
        output1 = self.projection1(output)
        output2 = self.projection2(output)
        
        return output1, output2