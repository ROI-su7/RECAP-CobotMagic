# 完整流程：从环境到 RECAP（Cobot Magic）

下面按执行顺序列出所有步骤。每个新终端先进入仓库并设置根目录：

```bash
cd ~/RECAP && export RECAP_ROOT=$PWD
```

```text
0 环境 ─▶ 1 CAN/相机 ─▶ 2 示教采集 ─▶ 3 转 LeRobot ─▶ 4 π0.5 全参微调 ─▶ 5 π0.5 真机推理
      ─▶ 6 带人工接管采集 rollout ─▶ 7 导出 ─▶ 8 模型资产 ─▶ 9 RECAP 四步训练 ─▶ 10 部署 CFG 策略 ─▶ 回到 6
```

更细的说明：机械臂一侧看 [FOUR_CAN.md](FOUR_CAN.md)，RECAP 的打分规则、参数含义和问题记录看 [openpi/RLinf/README_PIPER_RECAP.md](../openpi/RLinf/README_PIPER_RECAP.md)。

## 0. 环境

```bash
git clone https://github.com/ROI-su7/RECAP-CobotMagic.git ~/RECAP && cd ~/RECAP
bash env/install_all.sh
bash env/verify.sh
```

只在训练服务器上（不连机器人）：`bash env/install_all.sh --no-ros`、`bash env/verify.sh --no-ros`。

`06_recap_assets` 下载的 SigLIP2 和 Gemma 3 是 Google 的 gated 模型：先在 Hugging Face 网页接受许可，再执行 `openpi/RLinf/.venv-recap/bin/huggingface-cli login`，然后 `bash env/install_all.sh --only 06`。不要把 token 写进任何脚本或文档。

| 环境 | 位置 | 用途 |
|---|---|---|
| openpi | `openpi/.venv` | π0.5 微调、归一化统计、数据转换 |
| RECAP | `openpi/RLinf/.venv-recap` | 原生 JAX 推理服务、RECAP 训练与服务、真机客户端、记录器 |
| 系统 Python 3.8 | `/usr/bin/python3` | Piper ROS 节点、示教录制 `new_collect_data.py` |

## 1. CAN 与相机

**换一台机器必须先改 CAN 的 USB 映射和相机序列号**，见 [FOUR_CAN.md 第 1.3 节](FOUR_CAN.md)。每次开机：

```bash
# 终端 1
source /opt/ros/noetic/setup.bash && roscore
# 终端 2：只能在所有 Piper 节点都停止时执行
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic && sudo bash can_config.sh
ip -br link show type can     # can_l_leader / can_l_follower / can_r_leader / can_r_follower 都要 UP
# 终端 3
cd $RECAP_ROOT/cobot_magic/camera_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

## 2. 示教采集

固定 MASTER 的配置、预检、launch 和录制命令见 [FOUR_CAN.md 第 5 节](FOUR_CAN.md)。录制结果是 `~/recap_data_raw/hdf5/connect_strings/episode_*.hdf5`（30 fps，三路相机，14 维）。我们用的是 350 条成功的完整示教。采集结束后按第 5.5 节把主臂恢复成 OUTPUT。

## 3. 转换为 LeRobot

```bash
openpi/.venv/bin/python cobot_magic/collect_data/hdf5_lerobot.py \
  --datasetDir ~/recap_data_raw/hdf5/connect_strings \
  --datasetName local/all_wire_data \
  --targetDir openpi/lerobot_data/all_wire_data \
  --instruction "connect the light strings and the battery box to the splitter cable so the lights turn on."
```

`--targetDir` 会被重建，不要指向已有数据。提示词必须和 `openpi/src/openpi/training/config.py` 里 `pi05_connect_strings` 的 `default_prompt` 一致。

## 4. π0.5 全参微调

```bash
cd openpi
.venv/bin/python scripts/compute_lerobot_parquet_norm_stats.py --dataset-root lerobot_data/all_wire_data \
  --output-dir assets/pi05_connect_strings/all_wire_data --action-horizon 50
```

全参微调需要多卡（我们用 8×A100，训练 30000 步）。Slurm 模板 `openpi/slurm/train_pi05_connect_strings_8xa100.sbatch`：

```bash
mkdir -p logs && sbatch slurm/train_pi05_connect_strings_8xa100.sbatch
```

产物 `openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_repro/30000`。后面的脚本默认读这个路径；实验名或步数不同就设置 `RECAP_PI05_JAX_DIR`。在服务器上训练时，把这个目录 rsync 回机器人。

## 5. π0.5 真机推理

```bash
# 终端 A：四臂（可随时人工接管）或两从臂，二选一
bash openpi/RLinf/examples/recap_piper/run.sh launch takeover
# bash openpi/RLinf/examples/recap_piper/run.sh launch followers
# 终端 B：π0.5 服务（端口 6016）
bash openpi/RLinf/examples/recap_piper/run_native_server.sh
# 终端 D：真机客户端，profile 要和终端 A 一致
bash openpi/RLinf/examples/recap_piper/run_client.sh takeover
```

键盘用 `rosrun piper piper_mode_keyboard.py _profile:=inference`（两从臂用 `_profile:=follower`），或者直接用下一步的记录器。客户端启动后自动使能并开始推理，不要另外发布 `/enable_flag=true`。

## 6. 带人工接管采集 rollout

终端 A、B、D 同第 5 步，终端 C 是唯一的键盘和记录器（不要同时开 `piper_mode_keyboard.py`）：

```bash
cd $RECAP_ROOT/openpi/RLinf
export RECAP_RAW_DIR=$PWD/recap_data/raw_round001    # 每一轮单独一个目录
bash examples/recap_piper/run.sh collect --profile takeover
```

| 按键 | 作用 |
|---|---|
| I | 开始或继续回合 |
| T | 人工接管，看到 TAKEOVER ACTIVE 后再动主臂；纠正完按 I 交还模型，同一回合继续 |
| C / X | 三根全部插好 / 救不回来，打分并结束回合 |
| Space / H | HOLD / HOLD 后回原点（成功后必须先 H，重置线缆再按 I） |

接管是纠正，不是失败。每一轮按 C/X 打分的回合才会保存。

## 7. 导出 LeRobot 快照

```bash
cd $RECAP_ROOT/openpi/RLinf
export RECAP_RAW_DIR=$PWD/recap_data/raw_round001 RECAP_DATASET=$PWD/recap_data/lerobot_v1
bash examples/recap_piper/run.sh score     # 查看回合统计
bash examples/recap_piper/run.sh export
```

已有快照一律拒绝覆盖，下一轮用 `lerobot_v2`。

把 350 条示教做成隔离副本（`compute_returns` 会改写 `meta/`）：

```bash
SRC=$RECAP_ROOT/openpi/lerobot_data/all_wire_data; export RECAP_SFT_DATASET=$PWD/recap_data/sft_all_wire_data
mkdir -p "$RECAP_SFT_DATASET" && cp -al "$SRC/data" "$SRC/videos" "$RECAP_SFT_DATASET/" && cp -a "$SRC/meta" "$RECAP_SFT_DATASET/"
```

## 8. 模型资产

把微调好的 JAX checkpoint 转成 PyTorch（CPU 运行，内存大，建议在服务器上做）：

```bash
bash examples/recap_piper/run.sh convert
```

输出 `recap_artifacts/pi05_base`。SigLIP2 和 Gemma 3 由 `env/install_all.sh --only 06` 准备。`bash examples/recap_piper/run.sh check` 会检查这些资产是否齐全。

## 9. RECAP 四步训练

顺序固定：returns → Value → advantages → CFG（LoRA）。

**在 8 卡 Slurm 服务器上（推荐，我们用 8×A100）**：把代码和数据同步过去后，在登录节点运行：

```bash
cd ~/RECAP/openpi/RLinf
export RECAP_ROUND=round_001 RECAP_DATASET=$PWD/recap_data/lerobot_v1
mkdir -p recap_runs/$RECAP_ROUND && touch recap_runs/$RECAP_ROUND/.inputs_ready
P=$(sbatch --parsable --export=ALL slurm/recap_round_prep.sbatch)
sbatch --export=ALL --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch
bash slurm/recap_status.sh      # 查看进度
```

脚本会用试跑自动选能放进显存的 micro batch，全局批量固定（Value 256、CFG 128）；中断后重交会跳过已完成阶段并从最新 checkpoint 续训。参考用时（8×A100）：Value 8000 步约 4 小时，CFG 30000 步约 17 小时。多轮融合时用 `RECAP_ROLLOUT_DATASETS="路径1 路径2 ..."`。

CFG 不能用梯度累积：必须保持 全局批量 = micro batch × GPU 数（8 卡用默认的 128 = 16 × 8）。显存放不下 16 时，同时设置 `CFG_MICRO_CANDIDATES` 和 `CFG_GLOBAL_BATCH`（例如 8 卡 3090：`CFG_MICRO_CANDIDATES=8 CFG_GLOBAL_BATCH=64`），否则会报 `Cannot writeback when the parameter shape changes`。

**只有一张 4090**：按 [README_PIPER_RECAP.md 第 8 节](../openpi/RLinf/README_PIPER_RECAP.md) 手动逐步执行，四步都用 `examples/recap_piper/run.sh`。

取回策略：

```bash
R=round_001 STEP=30000 S=recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP/actor
mkdir -p $S && rsync -a --exclude dcp_checkpoint <server>:RECAP/openpi/RLinf/$S/ $S/
```

## 10. 部署 CFG 策略并进入下一轮

```bash
# 终端 A：四臂或两从臂（同第 5 步）
# 终端 B：CFG 服务，取代第 5 步的 run_native_server.sh
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run.sh serve \
  --checkpoint $PWD/recap_runs/round_001/cfg/piper_recap_cfg/checkpoints/global_step_30000 \
  --guidance-scale 1.0
# 终端 D：CFG 推理比 JAX 慢，预取改成 5 步
bash examples/recap_piper/run_client.sh takeover --rtc-prefix-len 5
```

终端 C 把 `RECAP_RAW_DIR` 换成下一轮的目录，回到第 6 步继续。

评估策略本身时，存到单独的目录，全程不按 T，`run.sh score` 统计的就是无接管成功率：

```bash
export RECAP_RAW_DIR=$PWD/recap_data/raw_round001_eval
bash examples/recap_piper/run.sh collect --profile followers   # 终端 A 用 launch followers，终端 D 用 followers
```
