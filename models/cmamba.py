import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

# take it from https://github.com/zclzcl0223/CMamba/blob/main/models/CMamba.py

def npo2(len):
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : [bs * nvars, d_ff, patch_num, d_state] or [B, D, L, N]
        # X : [bs * nvars, d_ff, patch_num, d_state]

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A : [bs * nvars, d_ff, patch_num, d_state] or [B, D, L, N]
        # X : [bs * nvars, d_ff, patch_num, d_state]

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
                    
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : [bs * nvars, patch_num, d_ff, d_state]
            X_in : [bs * nvars, patch_num, d_ff, d_state]

        Returns:
            H : [bs * nvars, patch_num, d_ff, d_state]
        """

        L = X_in.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in) # [bs * nvars, npo2(patch_num), d_ff, d_state]
            X = pad_npo2(X_in) # [bs * nvars, npo2(patch_num), d_ff, d_state]
        
        A = A.transpose(2, 1) # [bs * nvars, d_ff, npo2(patch_num), d_state]
        X = X.transpose(2, 1) # [bs * nvars, d_ff, npo2(patch_num), d_state]

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : [bs * nvars, patch_num, d_ff, d_state], X : [bs * nvars, d_ff, patch_num, d_state]
            grad_output_in : [bs * nvars, patch_num, d_ff, d_state]

        Returns:
            gradA : [bs * nvars, patch_num, d_ff, d_state], gradX : [bs * nvars, patch_num, d_ff, d_state]
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in) # [bs * nvars, npo2(patch_num), d_ff, d_state]
            A_in = pad_npo2(A_in) # [bs * nvars, npo2(patch_num), d_ff, d_state]

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1) # [bs * nvars, d_ff, npo2(patch_num), d_state]
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # [bs * nvars, d_ff, npo2(patch_num), d_state] shift 1 to the left (see hand derivation)

        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply

class GDDMLP(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc_sc = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                               nn.GELU(),
                               nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.fc_sf = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                               nn.GELU(),
                               nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.sigmoid = nn.Sigmoid()

        #self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fc_sc:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

        for layer in self.fc_sf:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

    def forward(self, x):
        b, n, p, d = x.shape
        scale = torch.zeros_like(x)
        shift = torch.zeros_like(x)
        if self.avg_flag:
            sc = self.fc_sc(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        if self.max_flag:
            sc = self.fc_sc(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        return self.sigmoid(scale) * x + self.sigmoid(shift)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars

class CMambaEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.layers = nn.ModuleList([CMambaBlock(configs) for _ in range(configs.e_layers)])

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]

        for layer in self.layers:
            x = layer(x)

        x = F.silu(x)

        return x

class CMambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.mixer = MambaBlock(configs)
        self.norm = RMSNorm(configs.d_model)

        self.gddmlp = configs.gddmlp
        if self.gddmlp:
            print("Insert GDDMLP")
            self.GDDMLP = GDDMLP(configs.c_out, configs.reduction, 
                                                 configs.avg, configs.max)
        self.dropout = nn.Dropout(configs.dropout)
        self.configs = configs

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]
        # output : [bs * nvars, patch_num, d_model]

        output = self.mixer(self.norm(x)) 

        if self.gddmlp:
            # output : [bs, nvars, patch_num, d_model]
            output = self.GDDMLP(output.reshape(-1, self.configs.c_out, 
                                                          output.shape[-2], output.shape[-1]))
            # output : [bs * nvars, patch_num, d_model]
            output = output.reshape(-1, output.shape[-2], output.shape[-1])
        output = self.dropout(output)
        output += x
        return output

class MambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(configs.d_model, 2 * configs.d_ff, bias=configs.bias)
        
        # projects x to input-dependent Δ, B, C, D
        self.x_proj = nn.Linear(configs.d_ff, configs.dt_rank + 2 * configs.d_state + configs.d_ff, bias=False)

        # projects Δ from dt_rank to d_ff
        self.dt_proj = nn.Linear(configs.dt_rank, configs.d_ff, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = configs.dt_rank**-0.5 * configs.dt_scale
        if configs.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif configs.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(configs.d_ff) * (math.log(configs.dt_max) - math.log(configs.dt_min)) + math.log(configs.dt_min)
        ).clamp(min=configs.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, configs.d_state + 1, dtype=torch.float32).unsqueeze(0)

        self.A_log = nn.Parameter(torch.log(A))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(configs.d_ff, configs.d_model, bias=configs.bias)

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]
        # y : [bs * nvars, patch_num, d_model]

        _, L, _ = x.shape

        xz = self.in_proj(x) # [bs * nvars, patch_num, 2 * d_ff]
        x, z = xz.chunk(2, dim=-1) # [bs * nvars, patch_num, d_ff], [bs * nvars, patch_num, d_ff]

        # x branch
        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # [bs * nvars, patch_num, d_ff]

        return output
    
    def ssm(self, x):
        # x : [bs * nvars, patch_num, d_ff]
        # y : [bs * nvars, patch_num, d_ff]

        A = -torch.exp(self.A_log.float()) # [d_ff, d_state]

        deltaBCD = self.x_proj(x) # [bs * nvars, patch_num, dt_rank + 2 * d_state + d_ff]
        # [bs * nvars, patch_num, dt_rank], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.configs.dt_rank, self.configs.d_state, self.configs.d_state, self.configs.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # [bs * nvars, patch_num, d_ff]

        if self.configs.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]
        # y : [bs * nvars, patch_num, d_ff]

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]
        
        hs = pscan(deltaA, BX)
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]
        # y : [bs * nvars, patch_num, d_ff]

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]

        h = torch.zeros(x.size(0), self.configs.d_ff, self.configs.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # [bs * nvars, patch_num, d_ff, d_state]
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class TSClassModel(nn.Module):
    """
    Multivariate time-series classification with CMamba encoder.
    Input : (bs, 1, seq_len, channels)
    Output: (bs, num_classes)
    """
    def __init__(
        self,
        seq_len: int,
        channels: int,
        num_classes: int,
        d_model: int = 64,
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 4,
        d_ff: int = 128,
        d_state: int = 16,
        dt_rank: int = 32,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        pscan: bool = True,
        bias: bool = True,
        dt_init: str = "constant",
        dt_scale: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        gddmlp: bool = False,
        reduction: int = 2,
        avg: bool = True,
        max_pool: bool = True,
    ):
        super().__init__()

        padding = stride
        self.patch_num = int((seq_len - patch_len) / stride + 2)
        self.patch_embed = PatchEmbedding(
            d_model, patch_len, stride, padding, head_dropout
        )

        self.encoder = CMambaEncoder(
            d_model=d_model,
            d_ff=d_ff,
            d_state=d_state,
            dt_rank=dt_rank,
            e_layers=e_layers,
            dropout=dropout,
            pscan=pscan,
            bias=bias,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            patch_num=self.patch_num,
            gddmlp=gddmlp,
            reduction=reduction,
            avg=avg,
            max_pool=max_pool,
            c_out=channels
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        bs, _, _, n_vars = x.shape
        x = x.squeeze(1).permute(0, 3, 2)

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std

        x, n_vars = self.patch_embed(x)
        x = self.encoder(x)

        x = x.view(bs, n_vars, x.size(1), x.size(2)).permute(0, 1, 3, 2)

        x = self.pool(x).squeeze(-1)
        x = x.mean(1)

        return self.classifier(x)