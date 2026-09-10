"""PLAN F8 - the single-GPU pin and the effective-batch arithmetic, without a GPU.

F8 is the defect most likely to silently invalidate the headline comparison, and
it is pure arithmetic: HF `Trainer` multiplies the per-device batch by the device
count, so "batch 16" on Kaggle's 2xT4 is really 32. These tests run in the local
environment, which has no torch (PLAN F7), because the bug does not need one.
"""

from __future__ import annotations

import os

import pytest

from src.utils import derive_per_device_batch, pin_single_gpu


@pytest.mark.parametrize("effective,devices,expected", [
    (16, 2, 8),     # the PRD recipe on Kaggle's 2xT4
    (16, 1, 16),    # the same recipe on one GPU
    (16, 0, 16),    # CPU-only: max(0, 1) devices
    (32, 2, 16),
])
def test_derive_per_device_batch(effective, devices, expected):
    assert derive_per_device_batch(effective, devices) == expected


def test_derive_per_device_batch_refuses_to_round():
    """Rounding here is exactly the F8 failure: the run would proceed at a batch
    size that no longer matches the recipe printed in the report."""
    with pytest.raises(ValueError, match="not divisible"):
        derive_per_device_batch(16, 3)


def test_effective_batch_is_recoverable_from_the_derived_pair():
    """What `runs.csv` records must multiply back to what was intended."""
    for devices in (1, 2, 4):
        per_device = derive_per_device_batch(16, devices)
        assert per_device * max(devices, 1) == 16


def test_pin_single_gpu_supplies_a_default(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    pin_single_gpu()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_pin_single_gpu_respects_an_explicit_choice(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=1` is someone picking the second card because the
    first is busy. The pin supplies a default; it does not overrule that."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    pin_single_gpu()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
