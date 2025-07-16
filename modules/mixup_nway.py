import torch
import numpy as np

class NWayMixupLoss:
    def __init__(self,
                 main_criterion,
                 hybrid_criterions,
                 seq_type_criterion,
                 orientation_criterion):
        self.main_criterion        = main_criterion
        self.hybrid_criterions     = hybrid_criterions
        self.seq_type_criterion    = seq_type_criterion
        self.orientation_criterion = orientation_criterion

    def __call__(self,
                 outputs,
                 seq_type_outputs,
                 orientation_outputs,
                 ext1_out1, ext2_out1, ext3_out1,
                 ext4_out1, ext5_out1, ext6_out1,
                 targets_list,
                 seq_type_targets_list,
                 orientation_targets_list,
                 lam):
        main_loss        = sum(l * self.main_criterion(outputs,                t)
                               for l, t in zip(lam, targets_list))

        ext1_out1_loss   = sum(l * self.hybrid_criterions[0](ext1_out1,        t)
                               for l, t in zip(lam, targets_list))
        ext2_out1_loss   = sum(l * self.hybrid_criterions[1](ext2_out1,        t)
                               for l, t in zip(lam, targets_list))
        ext3_out1_loss   = sum(l * self.hybrid_criterions[2](ext3_out1,        t)
                               for l, t in zip(lam, targets_list))
        ext4_out1_loss   = sum(l * self.hybrid_criterions[3](ext4_out1,        t)
                               for l, t in zip(lam, targets_list))
        ext5_out1_loss   = sum(l * self.hybrid_criterions[4](ext5_out1,        t)
                               for l, t in zip(lam, targets_list))
        ext6_out1_loss   = sum(l * self.hybrid_criterions[5](ext6_out1,        t)
                               for l, t in zip(lam, targets_list))

        seq_type_loss    = sum(l * self.seq_type_criterion(seq_type_outputs,   t)
                               for l, t in zip(lam, seq_type_targets_list))

        orientation_loss = sum(l * self.orientation_criterion(orientation_outputs, t)
                               for l, t in zip(lam, orientation_targets_list))

        return (main_loss, ext1_out1_loss, ext2_out1_loss, ext3_out1_loss,
                ext4_out1_loss, ext5_out1_loss, ext6_out1_loss,
                seq_type_loss, orientation_loss)

def mixup_batch_nway(batch, alpha=1.0, n=4, device='cuda'):
    lam_np = np.random.dirichlet([alpha] * n) if alpha > 0 else np.eye(1, n, 0)[0]
    lam    = torch.tensor(lam_np, dtype=torch.float32, device=device)

    B = batch['main_target'].size(0)

    perms = [torch.arange(B, device=device)]
    perms += [torch.randperm(B, device=device) for _ in range(n - 1)]

    target_keys = {'main_target', 'seq_type_aux_target', 'orientation_aux_target'}

    mixed_batch = {}
    for key, tensor in batch.items():
        if key not in target_keys:
            mixed_batch[key] = sum(lam[i] * tensor[perms[i]] for i in range(n))
        else:
            mixed_batch[key] = tensor

    targets_list            = [batch['main_target'][p]         for p in perms]
    seq_type_targets_list   = [batch['seq_type_aux_target'][p] for p in perms]
    orientation_targets_list= [batch['orientation_aux_target'][p] for p in perms]

    return (mixed_batch,
            targets_list,
            seq_type_targets_list,
            orientation_targets_list,
            lam)

def mixup_batch_by_category_nway(batch, alpha=1.0, n=4, device='cuda'):
    lam_np = np.random.dirichlet([alpha] * n) if alpha > 0 else np.eye(1, n, 0)[0]
    lam    = torch.tensor(lam_np, dtype=torch.float32, device=device)

    targets = batch['main_target']
    B       = targets.size(0)

    cat1_idx = torch.where(targets <= 7)[0]
    cat2_idx = torch.where(targets > 7)[0]

    def make_perms(idxs):
        if len(idxs) == 0:
            return []
        base = idxs
        add  = [idxs[torch.randperm(len(idxs), device=device)] for _ in range(n - 1)]
        return [base] + add

    perms_cat1 = make_perms(cat1_idx)
    perms_cat2 = make_perms(cat2_idx)

    perms = [torch.empty(B, dtype=torch.long, device=device) for _ in range(n)]
    for j in range(n):
        if len(cat1_idx): perms[j][cat1_idx] = perms_cat1[j]
        if len(cat2_idx): perms[j][cat2_idx] = perms_cat2[j]

    target_keys = {'main_target', 'seq_type_aux_target', 'orientation_aux_target'}
    mixed_batch = {}
    for key, tensor in batch.items():
        if key not in target_keys:
            mixed_batch[key] = sum(lam[i] * tensor[perms[i]] for i in range(n))
        else:
            mixed_batch[key] = tensor

    targets_list             = [batch['main_target'][p]           for p in perms]
    seq_type_targets_list    = [batch['seq_type_aux_target'][p]   for p in perms]
    orientation_targets_list = [batch['orientation_aux_target'][p] for p in perms]

    return (mixed_batch,
            targets_list,
            seq_type_targets_list,
            orientation_targets_list,
            lam)
