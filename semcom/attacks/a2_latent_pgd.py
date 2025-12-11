"""
Attack semantic representation S_enc.
"""

import torch
import torch.nn.functional as F
from .base_attack import BaseAttack
from .registry import register_attack

@register_attack
class LatentPGD(BaseAttack):
    name = "latent-pgd"

    def attack(self, model, x, y):
        model.eval()
        with torch.no_grad():
            s = model.encode(x)

        s_adv = s.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            out = model.decode(s_adv)
            loss = F.cross_entropy(out, y)
            loss.backward()

            grad = s_adv.grad.sign()
            s_adv = s_adv + self.eps * grad
            s_adv = s_adv.detach().requires_grad_(True)

        return s, s_adv
