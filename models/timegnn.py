import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import dense_to_sparse
import numpy as np

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

class TimeGNN(nn.Module):
    def __init__(self,
                 loss,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 seq_len,
                 aggregate="last",
                 keep_self_loops=False,
                 enforce_consecutive=False,
                 block_size=3):

        super().__init__()
        self.loss = loss
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.aggregate = aggregate
        self.keep_self_loops = keep_self_loops
        self.enforce_consecutive = enforce_consecutive

        self.conv11 = nn.Conv1d(input_dim, hidden_dim, 1, padding="same")
        self.conv12 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding="same", dilation=3)

        self.conv21 = nn.Conv1d(input_dim, hidden_dim, 1, padding="same")
        self.conv22 = nn.Conv1d(hidden_dim, hidden_dim, 5, padding="same", dilation=5)

        self.conv31 = nn.Conv1d(input_dim, hidden_dim, 1, padding="same")

        self.fc_final = nn.Linear(hidden_dim * 3, hidden_dim)

        ones = np.ones([seq_len, seq_len])
        self.rec_idx = torch.tensor(encode_onehot(np.where(ones)[0]), dtype=torch.float)
        self.send_idx = torch.tensor(encode_onehot(np.where(ones)[1]), dtype=torch.float)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.tri_mask = torch.tensor(np.tril(ones, k=-1)).bool()
        self.diagonal = torch.tensor(np.diag(np.diag(ones))).bool()
        self.consecutive = torch.tensor(np.eye(seq_len, k=1)).bool()

        self.gnns = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, normalize=False)
            for _ in range(block_size)
        ])
        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(block_size)])
        self.gnn_weights = nn.Linear(block_size, 1)

        self.fc_extra = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, return_graphs=False):
        x = x.squeeze(1) # (bs, seq_len, channels)
        x = x.permute(0, 2, 1) # (bs, channels, seq_len)

        bs = x.size(0)

        f1 = self.conv12(self.conv11(x))
        f2 = self.conv22(self.conv21(x))
        f3 = self.conv31(x)
        h = torch.cat([f1, f2, f3], dim=1)
        h = h.permute(0, 2, 1)
        h = F.relu(self.fc_final(h))

        recv = torch.bmm(self.rec_idx.repeat(bs, 1, 1), h)
        send = torch.matmul(self.send_idx, h)
        edges = self.edge_mlp(torch.cat([send, recv], dim=2))
        adj_logits = F.gumbel_softmax(edges, tau=0.5, hard=True)[:, :, 0]
        adj = adj_logits.view(bs, self.seq_len, self.seq_len)

        adj = adj.masked_fill(self.tri_mask, 0)
        if self.enforce_consecutive:
            adj = adj.masked_fill(self.consecutive, 1)
        if not self.keep_self_loops:
            adj = adj.masked_fill(self.diagonal, 0)

        edge_index, _ = dense_to_sparse(adj)
        h_flat = h.contiguous().view(-1, self.hidden_dim)

        gnn_outputs = []
        out = h_flat
        for gnn, bn in zip(self.gnns, self.bns):
            out = bn(gnn(out, edge_index))
            gnn_outputs.append(out)

        stacked = torch.stack(gnn_outputs, dim=-1)
        out = self.gnn_weights(stacked).squeeze(-1)
        out = F.relu(out).view(bs, self.seq_len, self.hidden_dim)

        out = out.mean(dim=1) if self.aggregate == "mean" else out[:, -1]
        out = F.relu(self.fc_extra(out))
        logits = self.out(out)

        return (logits, adj) if return_graphs else logits
