import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing

    def forward(self, logits, targets):
        nc = logits.shape[1]
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (nc-1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        ce = -(smooth * F.log_softmax(logits, dim=1)).sum(dim=1)
        p_t = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1-self.alpha)
        return (alpha_t * (1-p_t)**self.gamma * ce).mean()

def get_loss(cfg):
    return FocalLoss(cfg.training.loss.alpha, cfg.training.loss.gamma,
                     cfg.training.loss.label_smoothing)