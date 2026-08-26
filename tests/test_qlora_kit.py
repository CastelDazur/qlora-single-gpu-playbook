"""Offline tests for the guard logic. No GPU, no nvidia-smi needed."""

import json
import os

import pytest

from qlora_kit.vram import parse_nvidia_smi_free, choose_max_len, VramGateError
from qlora_kit.resume import remaining_steps, repair_trainer_state
from qlora_kit.preflight import check_alloc_conf
from qlora_kit.healthcheck import loss_trend_ok, run_healthcheck


# --- vram ---

def test_parse_nvidia_smi_free():
    assert parse_nvidia_smi_free("31000\n30500\n") == [31000, 30500]
    assert parse_nvidia_smi_free("31000 MiB\n") == [31000]   # tolerate stray unit
    assert parse_nvidia_smi_free("") == []


def test_choose_max_len_gates():
    assert choose_max_len(31000) == 2048
    assert choose_max_len(29000) == 1536
    assert choose_max_len(28000) == 1536      # exactly at the lower gate


def test_choose_max_len_aborts_when_too_low():
    with pytest.raises(VramGateError):
        choose_max_len(27000)


def test_choose_max_len_custom_gates():
    gates = [(20000, 4096), (10000, 2048)]
    assert choose_max_len(21000, gates=gates, abort_below=10000) == 4096
    assert choose_max_len(15000, gates=gates, abort_below=10000) == 2048


# --- resume ---

def test_remaining_steps():
    assert remaining_steps(742, 400) == 342
    assert remaining_steps(100, 100) == 0
    assert remaining_steps(100, 150) == 0      # never negative


def test_repair_trainer_state_adds_callbacks(tmp_path):
    p = tmp_path / "trainer_state.json"
    p.write_text(json.dumps({"global_step": 400}), encoding="utf-8")
    state = repair_trainer_state(p)
    assert "TrainerControl" in state["stateful_callbacks"]
    assert state["global_step"] == 400
    # It's valid JSON on disk now.
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    assert reloaded["stateful_callbacks"]["TrainerControl"]["args"]["should_training_stop"] is False


def test_repair_trainer_state_on_missing_file(tmp_path):
    state = repair_trainer_state(tmp_path, global_step=10)   # dir, no file yet
    assert state["global_step"] == 10
    assert (tmp_path / "trainer_state.json").exists()


# --- preflight ---

def test_check_alloc_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    ok, _ = check_alloc_conf()
    assert ok is True

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ok, msg = check_alloc_conf()
    assert ok is False and "485s" in msg

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    ok, _ = check_alloc_conf()
    assert ok is False


# --- healthcheck ---

def test_loss_trend_ok():
    assert loss_trend_ok([2.0, 1.8, 1.5, 1.2, 1.0]) is True
    assert loss_trend_ok([1.0, 1.0, 1.0, 1.0]) is False      # flat
    assert loss_trend_ok([1.0, 2.0]) is False                # too few points


def test_run_healthcheck_drives_step_fn():
    seq = iter([2.0, 1.7, 1.4, 1.1, 0.9, 0.8])
    ok, losses = run_healthcheck(lambda: next(seq), n_steps=6)
    assert ok is True and len(losses) == 6
