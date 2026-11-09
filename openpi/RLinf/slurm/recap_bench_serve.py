"""Time the RECAP CFG policy server's infer() on the local GPU with a real frame.

Usage (repo root, robot host, nothing else on the GPU):
  source examples/recap_piper/env.sh
  $RECAP_PYTHON slurm/recap_bench_serve.py <checkpoint dir> <raw episode dir> [n]
Builds the policy exactly as ``run.sh serve`` does and reports latency
percentiles, plus the plan for a fixed noise so two code versions can be
compared for identical output (``RECAP_BENCH_SAVE=<file.npy>``).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from rlinf.recap_piper.policy import PiperCFGPolicy, load_piper_cfg_model

checkpoint, episode = Path(sys.argv[1]), Path(sys.argv[2])
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 20
weights = checkpoint / "actor/model_state_dict/full_weights.pt"
with initialize_config_dir(
    config_dir=os.path.abspath("examples/offline_rl/config"), version_base="1.1"
):
    cfg = compose(config_name="piper_recap_cfg")
cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
model = load_piper_cfg_model(cfg.actor.model)
model.load_state_dict(
    torch.load(weights, map_location="cpu", weights_only=True), strict=True
)
object.__setattr__(model.config, "cfgrl_guidance_scale", 1.0)
policy = PiperCFGPolicy(model, task=os.environ["RECAP_TASK"])

frame = json.loads((episode / "frames.jsonl").open().readline())
obs = {
    "state": np.asarray(frame["state"], np.float32),
    "images": {
        cam: np.asarray(Image.open(episode / cam / "000000.png").convert("RGB"))
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    },
}

# Fixed noise for a reproducible plan across code versions.
original = model.sample_noise
noise = torch.Generator(device="cuda").manual_seed(0)
model.sample_noise = lambda shape, device: torch.randn(
    shape, generator=noise.manual_seed(0), device=device
)
for _ in range(3):
    policy.infer(obs)
times = []
for _ in range(repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    plan = policy.infer(obs)["actions"]
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
model.sample_noise = original
times = np.sort(times)
print(
    f"infer ms: p50 {np.percentile(times, 50):.1f}  p90 {np.percentile(times, 90):.1f}"
    f"  min {times[0]:.1f}  max {times[-1]:.1f}  (n={repeats})"
)
print("plan[0,:7]", np.round(plan[0, :7], 5))
if os.environ.get("RECAP_BENCH_SAVE"):
    np.save(os.environ["RECAP_BENCH_SAVE"], plan)
