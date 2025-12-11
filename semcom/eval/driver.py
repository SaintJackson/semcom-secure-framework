"""
Unified driver for:
- Running SemCom inference
- Applying attacks
- Computing accuracy
"""

import torch
from semcom.attacks.registry import get_attack
from semcom.config.defaults import CONFIG

def evaluate(model, dataloader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def run_attack(model, dataloader, attack_name, device="cuda"):
    attack = get_attack(attack_name)(eps=CONFIG["attack_eps"], steps=CONFIG["attack_steps"])
    acc_clean = evaluate(model, dataloader, device)

    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x_adv = attack.attack(model, x, y)
        if isinstance(x_adv, tuple):
            # latent attack returns (s, s_adv)
            s_adv = x_adv[1]
            out = model.decode(s_adv)
        else:
            _, out = model(x_adv)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc_adv = correct / total
    return acc_clean, acc_adv
