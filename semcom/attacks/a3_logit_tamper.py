"""
Corrupt final logits to mislead downstream tasks.
"""

import torch
from .base_attack import BaseAttack
from .registry import register_attack

@register_attack
class LogitTamper(BaseAttack):
    name = "logit-tamper"

    def attack(self, model, x, y):
        model.eval()
        with torch.no_grad():
            s, out = model(x)
        out_adv = out.clone()
        out_adv[:, :] = 0.0          # overwrite logits
        out_adv[:, torch.randint(0, out.size(1), (1,))] = 20.0  # force one class
        return out, out_adv
