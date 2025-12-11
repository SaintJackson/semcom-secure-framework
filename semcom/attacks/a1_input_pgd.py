"""
Adversarial attack on input x (PGD).
"""

import torch
import torch.nn.functional as F
from .base_attack import BaseAttack
from .registry import register_attack

@register_attack
class InputPGD(BaseAttack):
    name = "input-pgd"

    def attack(self, model, x, y):
        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            s, out = model(x_adv)
            loss = F.cross_entropy(out, y)
            loss.backward()

            grad = x_adv.grad.sign()
            x_adv = x_adv + self.eps * grad
            x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)

        return x_adv
