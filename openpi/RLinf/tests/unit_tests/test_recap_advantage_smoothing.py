"""Smoothed N-step RECAP advantages (rlinf.data.process.advantage)."""

import numpy as np
import pytest

from rlinf.data.process.advantage import smoothed_td_advantages, smoothing_window

RMIN, RMAX = -6481.0, 0.0


def _reference(returns, values, lookahead):
    """compute_advantages.py phase 2 for gamma = 1, one episode."""
    n = len(values)
    out = []
    for t in range(n):
        nxt = t + lookahead
        if nxt >= n:
            raw, v_next = returns[t], 0.0
        else:
            raw, v_next = returns[t] - returns[nxt], values[nxt]
        out.append((raw - RMIN) / (RMAX - RMIN) - 1.0 + v_next - values[t])
    return np.array(out)


def _episode(rng, n, success):
    rewards = -np.ones(n)
    rewards[-1] = 0.0 if success else -3500.0
    returns = np.cumsum(rewards[::-1])[::-1]
    values = returns / -RMIN + rng.normal(0, 0.02, n)
    return returns, values


def test_window_one_matches_unsmoothed_formula():
    rng = np.random.default_rng(0)
    r1, v1 = _episode(rng, 120, True)
    r2, v2 = _episode(rng, 37, False)
    ep = np.r_[np.zeros(120), np.ones(37)]
    adv, *_ = smoothed_td_advantages(
        ep,
        np.r_[r1, r2],
        np.r_[v1, v2],
        lookahead=10,
        window=1,
        return_min=RMIN,
        return_max=RMAX,
    )
    expected = np.r_[_reference(r1, v1, 10), _reference(r2, v2, 10)]
    np.testing.assert_allclose(adv, expected, rtol=0, atol=1e-12)


def test_smoothing_never_crosses_episodes():
    rng = np.random.default_rng(1)
    r1, v1 = _episode(rng, 60, True)
    r2, v2 = _episode(rng, 60, False)
    ep = np.r_[np.zeros(60), np.ones(60)]
    kwargs = {"lookahead": 50, "window": 15, "return_min": RMIN, "return_max": RMAX}
    base, *_ = smoothed_td_advantages(ep, np.r_[r1, r2], np.r_[v1, v2], **kwargs)
    changed, *_ = smoothed_td_advantages(
        ep, np.r_[r1, r2], np.r_[v1, v2 + 5.0], **kwargs
    )
    np.testing.assert_array_equal(base[:60], changed[:60])


def test_smoothing_reduces_label_flips():
    rng = np.random.default_rng(2)
    returns, values = _episode(rng, 2000, True)
    ep = np.zeros(2000)
    flips = []
    for window in (1, 15):
        adv, *_ = smoothed_td_advantages(
            ep,
            returns,
            values,
            lookahead=50,
            window=window,
            return_min=RMIN,
            return_max=RMAX,
        )
        label = adv >= np.percentile(adv, 70)
        flips.append(np.mean(label[1:] != label[:-1]))
    assert flips[1] < flips[0] / 3


def test_window_from_seconds():
    assert smoothing_window(1.0, 15) == 15
    assert smoothing_window(1.0, 30) == 31  # rounded up to odd: centred window
    assert smoothing_window(0.0, 15) == 1
    with pytest.raises(ValueError):
        smoothed_td_advantages(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            lookahead=1,
            window=2,
            return_min=RMIN,
            return_max=RMAX,
        )


def test_threshold_can_use_rollout_frames_only():
    from rlinf.data.process.advantage import threshold_scores

    sft, rollout = np.zeros(1000), np.linspace(-1.0, 1.0, 100)
    assert len(threshold_scores([sft, rollout], ["sft", "rollout"])) == 1100
    only = threshold_scores([sft, rollout], ["sft", "rollout"], ["rollout"])
    np.testing.assert_array_equal(only, rollout)
    with pytest.raises(ValueError):
        threshold_scores([sft], ["sft"], ["rollout"])


def test_piper_round2_advantage_and_cfg_settings(monkeypatch):
    from pathlib import Path

    import hydra

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    with hydra.initialize_config_dir(
        config_dir=str(root / "examples/offline_rl/config"), version_base="1.1"
    ):
        adv = hydra.compose(config_name="piper_recap_advantages")
        cfg = hydra.compose(config_name="piper_recap_cfg")
    assert adv.data.advantage_lookahead_step == 50
    assert adv.advantage.value_smoothing_s == 1.0
    assert adv.advantage.positive_quantile == 0.4
    assert list(adv.advantage.threshold_dataset_types) == ["rollout"]
    # pi*0.6 routing: explicit negative tag, 30% indicator dropout.
    assert cfg.actor.model.openpi.positive_only_conditional is False
    assert cfg.actor.model.openpi.unconditional_prob == 0.3


def test_full_finetune_trains_everything_and_keeps_lora_provenance(tmp_path):
    from omegaconf import OmegaConf
    from torch import nn

    from rlinf.recap_piper.data import atomic_json
    from rlinf.recap_piper.policy import (
        configure_trainable_parameters,
        policy_provenance,
    )

    model = nn.ModuleDict(
        {"gemma_expert": nn.Linear(2, 2), "paligemma": nn.Linear(2, 2)}
    )
    names = configure_trainable_parameters(model, lora_only=False, full=True)
    assert len(names) == 4 and all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):
        configure_trainable_parameters(model, lora_only=True, full=True)

    base = tmp_path / "base"
    base.mkdir()
    atomic_json(base / "config.json", {"converted_model_sha256": "x"})
    norm = tmp_path / "norm.json"
    atomic_json(norm, {})
    cfg = OmegaConf.create(
        {
            "model_path": str(base),
            "recap_lora_only": True,
            "openpi_data": {"norm_stats_path": str(norm), "default_prompt": "t"},
            "openpi": {"positive_only_conditional": True},
        }
    )
    # LoRA provenance is unchanged, so round_001 checkpoints still validate.
    assert "full_finetune" not in policy_provenance(cfg)
    cfg.recap_lora_only, cfg.recap_full_finetune = False, True
    assert policy_provenance(cfg)["full_finetune"] is True


def test_cfg_worker_method_kinds():
    """The advantage loader stays static; the model provider is an instance method."""
    import inspect

    from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker

    assert isinstance(
        inspect.getattr_static(FSDPCfgWorker, "_load_advantages_lookup"), staticmethod
    )
    assert not isinstance(
        inspect.getattr_static(FSDPCfgWorker, "model_provider_func"), staticmethod
    )
