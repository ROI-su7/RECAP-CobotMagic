"""CPU-only check of the RECAP Value / CFG data pipelines with the real configs.

Run on the login node before spending GPU time (repo root):
  source slurm/recap_server_env.sh
  $RECAP_PYTHON slurm/recap_check_loaders.py value "$VAL_DATA" data.train_num_workers=0 actor.micro_batch_size=4
  $RECAP_PYTHON slurm/recap_check_loaders.py cfg "$CFG_DATA" data.num_workers=0 +data.plain_sft=true actor.micro_batch_size=4
Builds the worker's dataloader exactly as training does (world size 1, no
CUDA, no Ray) and pulls a few batches through the collator.
"""

import logging
import os
import sys
import time

import torch
from hydra import compose, initialize_config_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
stage, overrides = sys.argv[1], sys.argv[2:]
name = {"value": "piper_recap_value", "cfg": "piper_recap_cfg"}[stage]
with initialize_config_dir(
    config_dir=os.path.abspath("examples/offline_rl/config"), version_base="1.1"
):
    cfg = compose(config_name=name, overrides=overrides)

if stage == "value":
    from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker as W
else:
    from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker as W

worker = W.__new__(W)
for key, value in {
    "cfg": cfg,
    "_world_size": 1,
    "_rank": 0,
    "logger": logging.getLogger(stage),
    "_logger": logging.getLogger(stage),
    "_stacklevel": 2,
}.items():
    try:
        object.__setattr__(worker, key, value)
    except AttributeError:  # read-only property on Worker
        pass

t = time.time()
result = worker.build_dataloader()
loader = result[0] if isinstance(result, tuple) else result
print(f"[{stage}] dataloader built in {time.time() - t:.1f}s: {type(loader).__name__}")


def describe(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        bad = "" if obj.dtype == torch.bool or torch.isfinite(obj.float()).all() else " NON-FINITE"
        return [f"{prefix}{tuple(obj.shape)} {obj.dtype}{bad}"]
    if isinstance(obj, dict):
        return [line for k, v in obj.items() for line in describe(v, f"{prefix}{k}.")]
    if isinstance(obj, (list, tuple)):
        return [line for i, v in enumerate(obj) for line in describe(v, f"{prefix}{i}.")]
    if hasattr(obj, "__dataclass_fields__"):
        return describe(vars(obj), prefix)
    return [f"{prefix}{type(obj).__name__}"]


it = iter(loader)
for i in range(int(os.environ.get("CHECK_BATCHES", "3"))):
    t = time.time()
    batch = next(it)
    print(f"[{stage}] batch {i} in {time.time() - t:.2f}s")
    if i == 0:
        print("\n".join("   " + line for line in describe(batch)))
print(f"[{stage}] OK")
