"""Compatibility names for the source-profile transaction controller."""
from arch_explorer.components.execution_profiles import NEUPIMS_LAYOUT
from arch_explorer.components.transaction_controller import Timing, Transaction, Bank, Channel

def switch_column_channel(address):
    return NEUPIMS_LAYOUT.stripe(address)

def decode(address):
    return NEUPIMS_LAYOUT.decode(address)
