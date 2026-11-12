"""Declarative model contracts. No graph, operator or timing implementations."""
from .pi05 import CONTRACT_ID, expand, normalize

def contract_id(kind):
    return CONTRACT_ID if kind == "pi05" else None
