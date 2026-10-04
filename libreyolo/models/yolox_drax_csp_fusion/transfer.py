"""Reproducible shared initialization for YOLOX-M comparison studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from ...utils.serialization import load_untrusted_torch_file


@dataclass(frozen=True)
class SharedTransferReport:
    source: str
    transferred_tensors: int
    transferred_parameters: int
    reset_seed: int
    excluded_prefixes: tuple[str, ...] = ("head.",)


class InitializeYOLOXSharedTransfer:
    """Load one YOLOX-M body into a control and candidate, then reset heads."""

    def __init__(self, source, control, candidate, *, reset_seed: int = 0):
        self.source = source
        self.control = self._network(control)
        self.candidate = self._network(candidate)
        self.reset_seed = int(reset_seed)

    @staticmethod
    def _network(value) -> nn.Module:
        network = getattr(value, "model", value)
        if not isinstance(network, nn.Module):
            raise TypeError("control and candidate must be torch modules or wrappers")
        return network

    def execute(self) -> SharedTransferReport:
        source_state, source_name = self._source_state()
        body_state = {
            key: value
            for key, value in source_state.items()
            if key.startswith("backbone.")
        }
        if not body_state:
            raise ValueError("source does not contain YOLOX backbone/PAN tensors")

        required = {key for key in self.control.state_dict() if key.startswith("backbone.")}
        missing = required - body_state.keys()
        if missing:
            raise ValueError(f"source is missing required YOLOX body tensors: {sorted(missing)[:5]}")

        self._validate_body(self.control, body_state, role="control")
        self._validate_body(self.candidate, body_state, role="candidate")
        self._load_body(self.control, body_state)
        self._load_body(self.candidate, body_state)
        self._reset_head(self.control.head, self.reset_seed)
        self._reset_head(self.candidate.head, self.reset_seed)
        return SharedTransferReport(
            source=source_name,
            transferred_tensors=len(body_state),
            transferred_parameters=sum(
                tensor.numel() for tensor in body_state.values()
            ),
            reset_seed=self.reset_seed,
        )

    def _source_state(self):
        source_name = "in-memory state dict"
        loaded = self.source
        if isinstance(loaded, (str, Path)):
            source_name = str(loaded)
            loaded = load_untrusted_torch_file(
                loaded, map_location="cpu", context="YOLOX shared-transfer weights"
            )
        if isinstance(loaded, nn.Module):
            loaded = loaded.state_dict()
        if isinstance(loaded, Mapping) and "model" in loaded:
            loaded = loaded["model"]
        if isinstance(loaded, nn.Module):
            loaded = loaded.state_dict()
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise TypeError("source must resolve to a state dict")
        state = dict(loaded)
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        if not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise TypeError("source state dict must map string keys to tensors")
        return state, source_name

    @staticmethod
    def _validate_body(network: nn.Module, body_state: dict, *, role: str):
        destination = network.state_dict()
        missing = [key for key in body_state if key not in destination]
        mismatched = [
            key
            for key, value in body_state.items()
            if key in destination and destination[key].shape != value.shape
        ]
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:5]}")
            if mismatched:
                details.append(f"shape mismatches: {mismatched[:5]}")
            joined_details = "; ".join(details)
            raise ValueError(
                f"{role} is not YOLOX-M body-compatible ({joined_details})"
            )

    @staticmethod
    def _load_body(network: nn.Module, body_state: dict):
        destination = network.state_dict()
        destination.update(body_state)
        network.load_state_dict(destination, strict=True)

    @staticmethod
    def _reset_head(head: nn.Module, seed: int):
        devices = sorted({
            parameter.device.index
            for parameter in head.parameters()
            if parameter.is_cuda
        })
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for module in head.modules():
                if module is not head and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        if hasattr(head, "initialize_biases"):
            head.initialize_biases(0.01)
