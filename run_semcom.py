"""
Main script for SemCom attack evaluation in plaintext setting.

This script:
- builds a sample SemCom model (ResNetSemCom)
- loads a placeholder CancerDataset (you can replace with Multi Cancer)
- runs A1/A2/A3 attacks in plaintext (no ZKP/FP)
- prints clean acc, attacked acc, ASR for each attack
"""

import torch
from torch.utils.data import DataLoader

from semcom.models.resnet_semcom import ResNetSemCom
from semcom.datasets.cancer_dataset import CancerDataset
from semcom.eval.plain_attack_exp import run_plain_attack_suite, eval_clean
from semcom.config.defaults import CONFIG


def main():
    device_str = CONFIG.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # TODO: 替换成你的 Multi Cancer dataset
    dataset_root = CONFIG.get("dataset_root", "./data")
    dataset = CancerDataset(dataset_root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = ResNetSemCom().to(device)

    # 先看干净语义通信的准确率
    clean_acc = eval_clean(model, loader, device)
    print(f"[Plain SemCom] Clean Top-1 accuracy: {clean_acc*100:.2f}%")

    # 跑三类攻击，都是在明文 SemCom（无证明层）下
    results = run_plain_attack_suite(model, loader, device)

    print("\n=== Plain-text SemCom Attack Results ===")
    print("Attack, CleanAcc, AttackedAcc, ASR")
    for attack_name, (ca, aa, asr) in results.items():
        print(f"{attack_name}, {ca:.4f}, {aa:.4f}, {asr:.4f}")


if __name__ == "__main__":
    main()
