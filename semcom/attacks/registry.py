"""
Attack registry for dynamic loading.
"""

ATTACKS = {}

def register_attack(cls):
    ATTACKS[cls.name] = cls
    return cls

def get_attack(name):
    return ATTACKS[name]
