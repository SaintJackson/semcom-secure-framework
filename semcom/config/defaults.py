"""
Default configuration for SemCom experiments.
"""

CONFIG = {
    "device": "cuda",
    "seed": 42,
    "quant_precision": 16,
    "attack_eps": 0.03,
    "attack_steps": 20,
    "dataset_root": "./data",
    "log_dir": "./logs"
}
