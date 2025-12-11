"""
Base class for SemCom attacks.
"""

class BaseAttack:
    name = "base-attack"

    def __init__(self, eps=0.03, steps=20):
        self.eps = eps
        self.steps = steps

    def attack(self, model, x, y):
        raise NotImplementedError
