import torch
from mamba_ssm import Mamba
import copy
from einops.layers.torch import Rearrange

class Original_TSCMamba(torch.nn.Module):
    def __init__(self,configs):
        super(Original_TSCMamba, self).__init__()
        self.configs=configs
        self.configs_copy = copy.deepcopy(self.configs)
        if self.configs.task_name=="classification":
            self.patcher=ConversionLayer(
                d_in=self.configs.enc_in,
                l_in=self.configs.seq_len,
                l_out=self.configs.projected_space,
                d_out=self.configs.enc_in,
                im_size=self.configs.rescale_size,
                patch_size=self.configs.patch_size)
            if self.configs.no_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space)
            elif self.configs.half_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space//2)

            self.ln=torch.nn.LayerNorm(self.configs.projected_space*3)
            self.dropout=torch.nn.Dropout(self.configs.dropout)
            self.learnable_focus=torch.nn.Parameter(torch.tensor([self.configs.initial_focus]))
            self.gelu=torch.nn.GELU()
            self.mamba1 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.projected_space*3,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.dconv,
                      expand=self.configs.e_fact) for _ in range(self.configs.num_mambas)
                ])
            self.mamba2 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.enc_in,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.dconv,
                      expand=self.configs.e_fact) for _ in range(self.configs.num_mambas)
                ])

            self.flatten=torch.nn.Flatten(start_dim=1)
            self.classifier=torch.nn.Sequential(
                torch.nn.Linear(self.configs.projected_space*3,(self.configs.projected_space*3)//2),
                torch.nn.Dropout(self.configs.dropout),
                torch.nn.Linear((self.configs.projected_space*3)//2,self.configs.num_class)        
            )
            
    def classification(self,x_cwt,batch_x_features):    
        x_patched=self.patcher(x_cwt)
        if self.configs.no_rocket==1:
            x_projected=self.projector(batch_x_features)
        else:
            if self.configs.half_rocket==0:
                x_projected=batch_x_features
            else:
                x_projected=torch.cat([batch_x_features[:,:,:self.configs.projected_space//2],
                                       self.projector(batch_x_features[:,:,self.configs.projected_space:])
                                    ],dim=2)
        if self.configs.additive_fusion==1:
            x_fused=(self.learnable_focus*x_projected)+(2.0-self.learnable_focus)*x_patched
        else:
            x_fused=(self.learnable_focus*x_projected)*(2.0-self.learnable_focus)*x_patched
        
        x_fused=self.gelu(x_fused)
        concatenated_x=torch.cat([x_patched,x_fused,x_projected],dim=2)
        concatenated_x=self.ln(concatenated_x)

        if self.configs.num_mambas!=0:
            x1=concatenated_x.clone()

            for i in range(self.configs.num_mambas):
                x1=self.mamba1[i](x1)+x1.clone()

            if self.configs.only_forward_scan==0:
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])
                x1_flipped=concatenated_x.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped=self.mamba1[i](x1_flipped)+x1_flipped.clone()     
                if self.configs.reverse_flip==0:           
                    x1=x1+x1_flipped
                elif self.configs.reverse_flip==1:
                    x1=x1+torch.flip(x1_flipped,dims=[self.configs.flip_dir])
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])

            concatenated_x=torch.permute(concatenated_x,(0,2,1))
            x2=concatenated_x.clone()
            for i in range(self.configs.num_mambas):
                x2=self.mamba2[i](x2)+x2.clone()
            x2=torch.permute(x2,(0,2,1))
            
            if self.configs.only_forward_scan==0:
                x2_flipped=torch.flip(concatenated_x.clone(),dims=[self.configs.flip_dir])
                for i in range(self.configs.num_mambas):
                    x2_flipped=self.mamba2[i](x2_flipped)+x2_flipped.clone()     

                x2_flipped=torch.permute(x2_flipped,(0,2,1))
                if self.configs.reverse_flip==0:
                    x2=x2+x2_flipped 
                elif self.configs.reverse_flip==1:
                    x2=x2+torch.flip(x2_flipped,dims=[self.configs.flip_dir])
            x3=x2+x1
        else:
            x3=concatenated_x
        if self.configs.max_pooling==0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)
        x3=self.flatten(x3)
        x_logits=self.classifier(x3)
        return x_logits

    def forward(self,x_cwt,x_features):
         if self.configs.task_name=='classification':
             return self.classification(x_cwt,x_features)
       
class ConversionLayer(torch.nn.Module):
    def __init__(self,d_in,l_in,d_out,l_out,im_size,patch_size):
        super(ConversionLayer, self).__init__()
        self.d_in=d_in
        self.l_in=l_in
        self.d_out=d_out
        self.l_out=l_out
        self.im_size=im_size
        self.patch_size=patch_size

        self.patch_embedding=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size), padding=0),
            Rearrange('b c h w -> b c (h w)')
        )
        self.projector=torch.nn.Linear((self.im_size//self.patch_size)*(self.im_size//self.patch_size),self.l_out)
        
    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.projector(x)
        return x
    
class TSCMamba_SingleSensor_v1(torch.nn.Module):
    def __init__(self, enc_in, seq_len, projected_space, rescale_size, patch_size, 
                 dropout, d_state, dconv, e_fact, num_mambas, num_class, 
                 only_forward_scan, flip_dir, reverse_flip, max_pooling):
        super(TSCMamba_SingleSensor_v1, self).__init__()
        
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.projected_space = projected_space
        self.rescale_size = rescale_size
        self.patch_size = patch_size
        self.dropout = dropout
        self.d_state = d_state
        self.dconv = dconv
        self.e_fact = e_fact
        self.num_mambas = num_mambas
        self.num_class = num_class
        self.only_forward_scan = only_forward_scan
        self.flip_dir = flip_dir
        self.reverse_flip = reverse_flip
        self.max_pooling = max_pooling
        
        # Patch embedding layer
        self.patcher = ConversionLayer(
            d_in=self.enc_in,
            l_in=self.seq_len,
            l_out=self.projected_space,
            d_out=self.enc_in,
            im_size=self.rescale_size,
            patch_size=self.patch_size
        )
        
        # Layer normalization for projected space
        self.ln = torch.nn.LayerNorm(self.projected_space)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.gelu = torch.nn.GELU()
        
        # Mamba blocks for sequence modeling
        self.mamba1 = torch.nn.ModuleList([
            Mamba(d_model=self.projected_space,
                  d_state=self.d_state,
                  d_conv=self.dconv,
                  expand=self.e_fact) for _ in range(self.num_mambas)
        ])
        
        self.mamba2 = torch.nn.ModuleList([
            Mamba(d_model=self.enc_in,
                  d_state=self.d_state,
                  d_conv=self.dconv,
                  expand=self.e_fact) for _ in range(self.num_mambas)
        ])
        
        # Classifier
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.projected_space, self.projected_space // 2),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.projected_space // 2, self.num_class)        
        )
    
    def forward(self, x_cwt):
        # Patch embedding
        x_patched = self.patcher(x_cwt)
        x_patched = self.gelu(x_patched)
        x_patched = self.ln(x_patched)
        
        # Mamba processing if enabled
        if self.num_mambas != 0:
            # First mamba processing
            x1 = x_patched.clone()
            for i in range(self.num_mambas):
                x1 = self.mamba1[i](x1) + x1.clone()
            
            # Bidirectional processing if enabled
            if self.only_forward_scan == 0:
                x_flipped = torch.flip(x_patched, dims=[self.flip_dir])
                x1_flipped = x_flipped.clone()
                for i in range(self.num_mambas):
                    x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
                
                if self.reverse_flip == 0:
                    x1 = x1 + x1_flipped
                elif self.reverse_flip == 1:
                    x1 = x1 + torch.flip(x1_flipped, dims=[self.flip_dir])
            
            # Second mamba processing (channel dimension)
            x_permuted = torch.permute(x1, (0, 2, 1))
            x2 = x_permuted.clone()
            for i in range(self.num_mambas):
                x2 = self.mamba2[i](x2) + x2.clone()
            
            # Bidirectional processing for second mamba
            if self.only_forward_scan == 0:
                x2_flipped = torch.flip(x_permuted.clone(), dims=[self.flip_dir])
                for i in range(self.num_mambas):
                    x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
                
                x2_flipped = torch.permute(x2_flipped, (0, 2, 1))
                if self.reverse_flip == 0:
                    x2 = torch.permute(x2, (0, 2, 1)) + x2_flipped
                elif self.reverse_flip == 1:
                    x2 = torch.permute(x2, (0, 2, 1)) + torch.flip(x2_flipped, dims=[self.flip_dir])
            else:
                x2 = torch.permute(x2, (0, 2, 1))
            
            x3 = x2 + x1
        else:
            x3 = x_patched
        
        # Global pooling
        if self.max_pooling == 0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)
        
        x3 = self.flatten(x3)
        x_logits = self.classifier(x3)
        return x_logits
    
class MultiSensor_TSCMamba_v1(torch.nn.Module):
    def __init__(self, enc_in_x, enc_in_y, enc_in_z, seq_len, projected_space, 
                 rescale_size, patch_size, dropout, d_state, dconv, e_fact, 
                 num_mambas, num_class, only_forward_scan, flip_dir, reverse_flip, max_pooling):
        super(MultiSensor_TSCMamba_v1, self).__init__()
        
        self.enc_in_x = enc_in_x
        self.enc_in_y = enc_in_y
        self.enc_in_z = enc_in_z
        self.seq_len = seq_len
        self.projected_space = projected_space
        self.rescale_size = rescale_size
        self.patch_size = patch_size
        self.dropout = dropout
        self.d_state = d_state
        self.dconv = dconv
        self.e_fact = e_fact
        self.num_mambas = num_mambas
        self.num_class = num_class
        self.only_forward_scan = only_forward_scan
        self.flip_dir = flip_dir
        self.reverse_flip = reverse_flip
        self.max_pooling = max_pooling
        
        # Separate patch embedding layers for each sensor
        self.patcher_x = ConversionLayer(
            d_in=self.enc_in_x,
            l_in=self.seq_len,
            l_out=self.projected_space,
            d_out=self.enc_in_x,
            im_size=self.rescale_size,
            patch_size=self.patch_size
        )
        
        self.patcher_y = ConversionLayer(
            d_in=self.enc_in_y,
            l_in=self.seq_len,
            l_out=self.projected_space,
            d_out=self.enc_in_y,
            im_size=self.rescale_size,
            patch_size=self.patch_size
        )
        
        self.patcher_z = ConversionLayer(
            d_in=self.enc_in_z,
            l_in=self.seq_len,
            l_out=self.projected_space,
            d_out=self.enc_in_z,
            im_size=self.rescale_size,
            patch_size=self.patch_size
        )
        
        # Cross-CWT processing layers
        self.cross_cwt_processor = torch.nn.Sequential(
            torch.nn.Conv1d(self.projected_space * 2, self.projected_space, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(self.projected_space),
            torch.nn.GELU()
        )
        
        # Fusion layer for combining all features (3 individual + 3 cross)
        total_features = self.projected_space * 6  # 3 individual + 3 cross
        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(total_features, self.projected_space),
            torch.nn.BatchNorm1d(self.projected_space),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout)
        )
        
        # Layer normalization for projected space
        self.ln = torch.nn.LayerNorm(self.projected_space)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.gelu = torch.nn.GELU()
        
        # Mamba blocks for sequence modeling
        self.mamba1 = torch.nn.ModuleList([
            Mamba(d_model=self.projected_space,
                  d_state=self.d_state,
                  d_conv=self.dconv,
                  expand=self.e_fact) for _ in range(self.num_mambas)
        ])
        
        # Note: Using projected_space instead of enc_in for unified processing
        self.mamba2 = torch.nn.ModuleList([
            Mamba(d_model=self.projected_space,
                  d_state=self.d_state,
                  d_conv=self.dconv,
                  expand=self.e_fact) for _ in range(self.num_mambas)
        ])
        
        # Classifier
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.projected_space, self.projected_space // 2),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.projected_space // 2, self.num_class)        
        )
    
    def compute_cross_cwt(self, x1, x2):
        """
        Compute cross-CWT between two signals
        x1, x2: [batch_size, projected_space, seq_len]
        """
        # Concatenate the two signals along the channel dimension
        combined = torch.cat([x1, x2], dim=1)  # [batch_size, projected_space*2, seq_len]
        
        # Process through cross-CWT processor
        cross_features = self.cross_cwt_processor(combined)  # [batch_size, projected_space, seq_len]
        
        return cross_features
    
    def forward(self, x_cwt, y_cwt, z_cwt):
        # Individual patch embeddings for each sensor
        x_patched = self.gelu(self.patcher_x(x_cwt))
        y_patched = self.gelu(self.patcher_y(y_cwt))
        z_patched = self.gelu(self.patcher_z(z_cwt))
        
        # Reshape for cross-CWT computation: [batch_size, projected_space, seq_len]
        x_reshaped = x_patched.transpose(1, 2)
        y_reshaped = y_patched.transpose(1, 2)
        z_reshaped = z_patched.transpose(1, 2)
        
        # Compute cross-CWT between all pairs
        cross_xy = self.compute_cross_cwt(x_reshaped, y_reshaped)
        cross_xz = self.compute_cross_cwt(x_reshaped, z_reshaped)
        cross_yz = self.compute_cross_cwt(y_reshaped, z_reshaped)
        
        # Global pooling for each feature set
        if self.max_pooling == 0:
            x_pooled = x_reshaped.mean(2)
            y_pooled = y_reshaped.mean(2)
            z_pooled = z_reshaped.mean(2)
            cross_xy_pooled = cross_xy.mean(2)
            cross_xz_pooled = cross_xz.mean(2)
            cross_yz_pooled = cross_yz.mean(2)
        else:
            x_pooled, _ = x_reshaped.max(2)
            y_pooled, _ = y_reshaped.max(2)
            z_pooled, _ = z_reshaped.max(2)
            cross_xy_pooled, _ = cross_xy.max(2)
            cross_xz_pooled, _ = cross_xz.max(2)
            cross_yz_pooled, _ = cross_yz.max(2)
        
        # Concatenate all features
        all_features = torch.cat([
            x_pooled, y_pooled, z_pooled,
            cross_xy_pooled, cross_xz_pooled, cross_yz_pooled
        ], dim=1)
        
        # Fusion layer
        fused_features = self.fusion_layer(all_features)
        
        # Reshape back for Mamba processing: [batch_size, seq_len, projected_space]
        # We'll use a dummy sequence length of 1 since we've already pooled
        x_fused = fused_features.unsqueeze(1)  # [batch_size, 1, projected_space]
        
        # Apply layer normalization
        x_fused = self.ln(x_fused)
        
        # Mamba processing if enabled
        if self.num_mambas != 0:
            # First mamba processing
            x1 = x_fused.clone()
            for i in range(self.num_mambas):
                x1 = self.mamba1[i](x1) + x1.clone()
            
            # Bidirectional processing if enabled
            if self.only_forward_scan == 0:
                x_flipped = torch.flip(x_fused, dims=[self.flip_dir])
                x1_flipped = x_flipped.clone()
                for i in range(self.num_mambas):
                    x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
                
                if self.reverse_flip == 0:
                    x1 = x1 + x1_flipped
                elif self.reverse_flip == 1:
                    x1 = x1 + torch.flip(x1_flipped, dims=[self.flip_dir])
            
            # Second mamba processing (channel dimension)
            x_permuted = torch.permute(x1, (0, 2, 1))
            x2 = x_permuted.clone()
            for i in range(self.num_mambas):
                x2 = self.mamba2[i](x2) + x2.clone()
            
            # Bidirectional processing for second mamba
            if self.only_forward_scan == 0:
                x2_flipped = torch.flip(x_permuted.clone(), dims=[self.flip_dir])
                for i in range(self.num_mambas):
                    x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
                
                x2_flipped = torch.permute(x2_flipped, (0, 2, 1))
                if self.reverse_flip == 0:
                    x2 = torch.permute(x2, (0, 2, 1)) + x2_flipped
                elif self.reverse_flip == 1:
                    x2 = torch.permute(x2, (0, 2, 1)) + torch.flip(x2_flipped, dims=[self.flip_dir])
            else:
                x2 = torch.permute(x2, (0, 2, 1))
            
            x3 = x2 + x1
        else:
            x3 = x_fused
        
        # Final pooling and classification
        x3 = x3.squeeze(1)  # Remove dummy sequence dimension
        x_logits = self.classifier(x3)
        return x_logits