import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def compute_components(self, predict, target, mask, weights=None):
        """Return known/hole losses normalized by their valid pixel areas."""
        if mask.shape[1] != predict.shape[1]:
            mask_3c = mask.expand(-1, predict.shape[1], -1, -1)
        else:
            mask_3c = mask
        mask_3c = mask_3c.to(dtype=predict.dtype, device=predict.device)
        hole_3c = 1 - mask_3c

        diff = self.loss_fn(predict, target, reduction='none')
        known_denom = mask_3c.sum(dim=(1, 2, 3)).clamp_min(1.0)
        hole_denom = hole_3c.sum(dim=(1, 2, 3)).clamp_min(1.0)

        lossu = (diff * mask_3c).sum(dim=(1, 2, 3)) / known_denom
        lossm = (diff * hole_3c).sum(dim=(1, 2, 3)) / hole_denom

        loss_hole_weighted = 10 * lossm
        loss = lossu + loss_hole_weighted
        if self.is_weighted and weights is not None:
            loss = weights * loss

        return {
            'loss_total': loss.mean(),
            'loss_known': lossu.mean(),
            'loss_hole': lossm.mean(),
            'loss_hole_weighted': loss_hole_weighted.mean(),
        }

    def forward(self, predict, target, mask, weights=None):
        return self.compute_components(predict, target, mask, weights)['loss_total']
      
