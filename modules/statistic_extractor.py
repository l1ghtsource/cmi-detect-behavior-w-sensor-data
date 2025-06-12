import torch
import torch.nn as nn

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output

def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

class StatisticExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, pad_mask=None):
        """
        Args:
            data: [B, C, L] - batch, channels, length
            pad_mask: [B, L] - padding mask (1=valid, 0=padding)
        Returns:
            stats: [B, C*10] - 10 statistics per channel
        """
        B, C, L = data.shape
        
        if pad_mask is not None:
            mask_expanded = pad_mask.unsqueeze(1).expand(-1, C, -1)  # [B, C, L]
            data = data.clone()
            data[~mask_expanded.bool()] = float('nan')
        
        mean_val = torch.nanmean(data, dim=2)  # [B, C]
        std_val = nanstd(data, dim=2)  # [B, C]
        max_val, _ = nanmax(data, dim=2)  # [B, C]
        min_val, _ = nanmin(data, dim=2)  # [B, C]
        
        # note: quantile doesn't handle NaN well, so we need to work around it
        q25_list = []
        q75_list = []
        
        for b in range(B):
            q25_batch = []
            q75_batch = []
            for c in range(C):
                series = data[b, c, :]
                valid_series = series[~torch.isnan(series)]
                
                if len(valid_series) > 0:
                    q25_val = torch.quantile(valid_series, 0.25)
                    q75_val = torch.quantile(valid_series, 0.75)
                else:
                    q25_val = torch.tensor(0.0, device=data.device)
                    q75_val = torch.tensor(0.0, device=data.device)
                
                q25_batch.append(q25_val)
                q75_batch.append(q75_val)
            
            q25_list.append(torch.stack(q25_batch))
            q75_list.append(torch.stack(q75_batch))
        
        q25 = torch.stack(q25_list)  # [B, C]
        q75 = torch.stack(q75_list)  # [B, C]

        diff = data[:, :, 1:] - data[:, :, :-1]  # [B, C, L-1]
        diff_mean = torch.nanmean(diff, dim=2)  # [B, C]
        diff_std = nanstd(diff, dim=2)  # [B, C]

        energy = torch.nansum(data ** 2, dim=2)  # [B, C]
        rms = torch.sqrt(torch.nanmean(data ** 2, dim=2))  # [B, C]

        stats = torch.cat([
            mean_val, std_val, max_val, min_val, q25, q75,
            diff_mean, diff_std, energy, rms
        ], dim=1)  # [B, C*10]

        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        return stats