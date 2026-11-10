"""The value-network layer (PLAN_valuenet.md, section F): ranges as equity-bin histograms,
the counterfactual value network, and NetLeaf, the network as a turn-end value provider.

Builds on core/ only. features.py is numpy only; model.py and leaf.py import torch and
data.py imports h5py, so import those modules only where torch / h5py are installed.
"""
