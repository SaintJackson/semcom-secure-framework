"""
Plain-text SemCom attack experiments.

Assumptions:
- model: BaseSemCom-like (encode -> decode -> logits)
- attacks: registered in semcom.attacks.registry
- dataset: classification-style (x, y)

This file provides:
- clean accuracy evaluation
- attacked accuracy
- attack success rate (ASR)
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semcom.attacks.registry import get_attack
from semcom.config.defaults import CONFIG


@torch.no_grad()
def eval_clean(model: nn.Module,
               dataloader: DataLoader,
               device: torch.device) -> float:
    """Evaluate clean (no-attack) Top-1 accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, out = model(x)
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples if total_samples > 0 else 0.0


def eval_attack_input(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    attack_name: str,
    eps: float,
    steps: int,
) -> Tuple[float, float, float]:
    """
    A1: 输入级攻击(Input PGD/FGSM)。

    返回:
        clean_acc      : 未攻击时整体 Top-1 accuracy
        attacked_acc   : 攻击后整体 Top-1 accuracy
        asr            : Attack Success Rate:
                         在“原本预测正确”的样本上，被攻击后变错的比例
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    attack_cls = get_attack(attack_name)
    attack = attack_cls(eps=eps, steps=steps)

    total_clean_correct = 0
    total_attacked_correct = 0
    total_initial_correct = 0
    total_success_attacks = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = y.size(0)
        total_samples += batch_size

        # clean prediction
        with torch.no_grad():
            _, out_clean = model(x)
            preds_clean = out_clean.argmax(dim=1)
            clean_correct_mask = (preds_clean == y)
            c_clean = clean_correct_mask.sum().item()
            total_clean_correct += c_clean

        # generate adversarial x
        x_adv = attack.attack(model, x, y)
        if isinstance(x_adv, tuple):  # 容错一下接口
            x_adv = x_adv[0]

        with torch.no_grad():
            _, out_adv = model(x_adv)
            preds_adv = out_adv.argmax(dim=1)

        total_attacked_correct += (preds_adv == y).sum().item()

        # ASR 只统计原本预测正确样本
        if c_clean > 0:
            total_initial_correct += c_clean
            success_mask = clean_correct_mask & (preds_adv != y)
            total_success_attacks += success_mask.sum().item()

    clean_acc = total_clean_correct / total_samples if total_samples > 0 else 0.0
    attacked_acc = total_attacked_correct / total_samples if total_samples > 0 else 0.0
    asr = (total_success_attacks / total_initial_correct) if total_initial_correct > 0 else 0.0
    return clean_acc, attacked_acc, asr


def eval_attack_latent(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    attack_name: str,
    eps: float,
    steps: int,
) -> Tuple[float, float, float]:
    """
    A2: 语义 latent 攻击 (LatentPGD)。
    attack.attack(model, x, y) 需返回 (s_clean, s_adv)。
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()  # 目前没用到，但可以扩展

    attack_cls = get_attack(attack_name)
    attack = attack_cls(eps=eps, steps=steps)

    total_clean_correct = 0
    total_attacked_correct = 0
    total_initial_correct = 0
    total_success_attacks = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = y.size(0)
        total_samples += batch_size

        # clean prediction
        with torch.no_grad():
            _, out_clean = model(x)
            preds_clean = out_clean.argmax(dim=1)
            clean_correct_mask = (preds_clean == y)
            c_clean = clean_correct_mask.sum().item()
            total_clean_correct += c_clean

        # latent attack
        s_clean, s_adv = attack.attack(model, x, y)
        with torch.no_grad():
            out_adv = model.decode(s_adv)
            preds_adv = out_adv.argmax(dim=1)

        total_attacked_correct += (preds_adv == y).sum().item()

        if c_clean > 0:
            total_initial_correct += c_clean
            success_mask = clean_correct_mask & (preds_adv != y)
            total_success_attacks += success_mask.sum().item()

    clean_acc = total_clean_correct / total_samples if total_samples > 0 else 0.0
    attacked_acc = total_attacked_correct / total_samples if total_samples > 0 else 0.0
    asr = (total_success_attacks / total_initial_correct) if total_initial_correct > 0 else 0.0
    return clean_acc, attacked_acc, asr


@torch.no_grad()
def eval_attack_task_tamper(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    attack_name: str,
) -> Tuple[float, float, float]:
    """
    A3: 任务层 logit 篡改 (LogitTamper)。

    attack.attack(model, x, y) -> (out_clean, out_adv)

    返回:
        clean_acc      : 未篡改时整体 Top-1 accuracy
        attacked_acc   : 篡改后整体 Top-1 accuracy
        asr            : 在 clean 预测正确样本上，被 tamper 之后变错的比例
    """
    model.eval()
    attack_cls = get_attack(attack_name)
    attack = attack_cls(eps=0.0, steps=0)

    total_clean_correct = 0
    total_attacked_correct = 0
    total_initial_correct = 0
    total_success_attacks = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = y.size(0)
        total_samples += batch_size

        out_clean, out_adv = attack.attack(model, x, y)
        preds_clean = out_clean.argmax(dim=1)
        preds_adv = out_adv.argmax(dim=1)

        clean_correct_mask = (preds_clean == y)
        c_clean = clean_correct_mask.sum().item()
        total_clean_correct += c_clean
        total_attacked_correct += (preds_adv == y).sum().item()

        if c_clean > 0:
            total_initial_correct += c_clean
            success_mask = clean_correct_mask & (preds_adv != y)
            total_success_attacks += success_mask.sum().item()

    clean_acc = total_clean_correct / total_samples if total_samples > 0 else 0.0
    attacked_acc = total_attacked_correct / total_samples if total_samples > 0 else 0.0
    asr = (total_success_attacks / total_initial_correct) if total_initial_correct > 0 else 0.0
    return clean_acc, attacked_acc, asr


def run_plain_attack_suite(model: nn.Module,
                           dataloader: DataLoader,
                           device: torch.device):
    """
    一次性跑完 A1 / A2 / A3 三类 attack。
    返回 dict: {attack_name: (clean_acc, attacked_acc, asr)}
    """
    eps = CONFIG.get("attack_eps", 0.03)
    steps = CONFIG.get("attack_steps", 20)

    results = {}

    # A1: 输入级 PGD
    ca1, aa1, asr1 = eval_attack_input(
        model, dataloader, device,
        attack_name="input-pgd", eps=eps, steps=steps
    )
    results["A1-input-pgd"] = (ca1, aa1, asr1)

    # A2: latent PGD
    ca2, aa2, asr2 = eval_attack_latent(
        model, dataloader, device,
        attack_name="latent-pgd", eps=eps, steps=steps
    )
    results["A2-latent-pgd"] = (ca2, aa2, asr2)

    # A3: logit tamper
    ca3, aa3, asr3 = eval_attack_task_tamper(
        model, dataloader, device,
        attack_name="logit-tamper"
    )
    results["A3-logit-tamper"] = (ca3, aa3, asr3)

    return results
