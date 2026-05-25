# WM Demo 在原 DA-PFC 模型上的扩展记录

> 参考基线：`outputs/exp_2026-04-14_10-26-10_resume_DA15nM_100s/`
> （N=1000 全网络稀疏均匀连接，单通道 5 ms 突触，DA 阶跃到 15 nM 的 resume 实验）
>
> WM Demo 入口：`experiments/exp_wm_demo.py`
> 内核：`models/kernels.py :: run_wm_kernel_dual`
> 网络：`models/network.py :: create_wm_network`

---

## 一、启动命令

### 1.1 前置：生成 2 nM 稳态 checkpoint（只需一次）

```bash
python main.py --da 2.0 --duration 100 --save-ckpt
# 产物：checkpoints/ckpt_DA2nM_bg200_100s.pkl
```

### 1.2 当前推荐启动命令（v10 / v11 工作记忆配置）

```bash
python experiments/exp_wm_demo.py \
  --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl \
  --da 2.0 --gpu 0 --tag wm_sfa_v10 \
  --pool-size 100 --intra-w 4.0 --intra-prob 0.5 \
  --cue-amp 300.0 --cue-ms 1500.0 \
  --delay-ms 5000.0 --baseline-ms 5000.0 --probe-ms 500.0 \
  --mem-bg-offset -10 \
  --sfa-b 0.8 --sfa-tau 2000 \
  --alpha-gate 0.15 \
  --iwm-size 60 --e2i-w 3.0 --e2i-prob 0.20 \
  --i2e-w -3.5 --i2e-prob 0.15
```

### 1.3 关闭 SFA 的对照运行

```bash
python experiments/exp_wm_demo.py \
  --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl \
  --da 2.0 --gpu 0 --tag wm_nosfa \
  --pool-size 100 --intra-w 4.0 --intra-prob 0.5 \
  --cue-amp 300.0 --cue-ms 1500.0 \
  --delay-ms 5000.0 --baseline-ms 5000.0 --probe-ms 500.0 \
  --mem-bg-offset -10 \
  --sfa-b 0 \
  --alpha-gate 0.15 \
  --iwm-size 60 --e2i-w 3.0 --e2i-prob 0.20 \
  --i2e-w -3.5 --i2e-prob 0.15
```

### 1.4 关闭 WTA 的对照运行

```bash
python experiments/exp_wm_demo.py \
  --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl \
  --da 2.0 --gpu 0 --tag wm_nowta \
  --pool-size 100 --intra-w 4.0 --intra-prob 0.5 \
  --cue-amp 300.0 \
  --mem-bg-offset -10 --sfa-b 0.8 --sfa-tau 2000 \
  --alpha-gate 0.15 \
  --no-wta
```

### 1.5 输出目录

```
outputs/exp_<timestamp>_<tag>_DA<da>nM/
├── config.json              # 本次运行使用的所有参数
├── raw_data.pkl             # 原始 spikes / V / α / I_total
├── combined_raster.png      # 项目标准 3×2 raster
├── combined_rates_all.png   # 项目标准 3×2 firing-rate（高斯平滑）
├── wm_overview.png          # WM 协议带（cue/delay/probe）+ Mem-A/B/E-BG 速率
├── wm_rates_pools.png       # 仅三个 pool 的速率对比
├── wm_report.txt            # 各窗口平均发放率、HIGH 状态判定
└── analysis_report.txt      # 标准 PFCAnalyzer 报告
```

---

## 二、相对原模型的所有新增设定

> **完全保留**：N_E=824 / N_I=176、D1/D2 mask、`CONN_PROB=0.02` 全网络背景连接、`W_EXC=3.0` / `W_INH=-20`、所有 LIF 参数、`BG_MEAN=200`、`DA→α(D1,D2)` 调控全链、checkpoint 指纹机制。
>
> WM Demo 是在这之上**叠加**的可关掉的扩展模块。

### 2.1 ① 选择性 E 子池：Mem-A / Mem-B（结构层）

从 `E-Other`（既不属于 D1 也不属于 D2 的兴奋池）切出两块各 100 神经元，**池内自循环密化**：

| 项目 | 原模型（E-Other 内部） | WM Demo（Mem 池内部） |
|------|----------------------|---------------------|
| 连接概率 | 0.02 | **0.5**（25 倍） |
| 突触权重 | 3.0 pA | **4.0 pA** |
| 跨池连接 | 同样的 0.02 | **0**（A↔B 完全独立） |

物理意义：吸引子骨架。没有它，cue 撤销后没有任何东西能维持高发放。

CLI：`--pool-size 100 --intra-prob 0.5 --intra-w 4.0`
代码：[models/network.py](../models/network.py) `create_wm_network`

---

### 2.2 ② 双通道突触：AMPA + NMDA（kernel 层最大改动）

原 kernel 只有一种 5 ms 突触。WM kernel `run_wm_kernel_dual` 把整张权重矩阵拆成两份：

| 通道 | τ | 谁走这条 |
|------|----|--------|
| AMPA / GABA-A | **5 ms** | 除 Mem 池内 E→E 之外的全部连边（E→I、I→E、I→I、跨池 E→E、Mem→I-WM、I-WM→Mem） |
| NMDA（饱和门控）| **100 ms** | **仅** Mem-A↔Mem-A、Mem-B↔Mem-B 池内 E→E |

NMDA 门控方程（Wang 2002）：
```
ds_j/dt = -s_j/τ_NMDA + α_gate · (1 - s_j) · spike_j
I_NMDA  = (s @ W_nmda_t)
```

物理意义：
- AMPA 5 ms → I-WM 反馈能即时跟上（WTA 真正生效）
- NMDA 100 ms 慢积分器 → 是"持续几秒不衰减"的物理来源
- `α_gate` 决定 NMDA 多快饱和，越小 → 双稳态越宽、对噪声越鲁棒

CLI：`--alpha-gate 0.15`（默认 0.5，调小后更稳）
代码：[models/kernels.py](../models/kernels.py) `run_wm_kernel_dual`，第 ~1300-1560 行

---

### 2.3 ③ WTA 共享抑制环（结构层）

从 `N_I=176` 中再切出 60 个组成 **I-WM** 子池，把 Mem ↔ I-WM 连接密化：

| 连边 | 概率 | 权重 |
|------|------|------|
| Mem-A,Mem-B → I-WM | 0.20 | +3.0 pA |
| I-WM → Mem-A,Mem-B | 0.15 | -3.5 pA |

预算（基线 Mem≈0 Hz / I-WM≈13 Hz）：
```
I_inhib = 60 · 0.15 · 3.5 · 0.1 · 13 ≈ 41 pA  →  ΔV ≈ -4 mV   (压住未中招的池)
```
Mem-A 赢出后 I-WM 升至 ~22 Hz，对 Mem-B 的抑制升到 ~70 pA / -7 mV，构成 winner-take-all。

CLI：`--iwm-size 60 --e2i-w 3.0 --e2i-prob 0.20 --i2e-w -3.5 --i2e-prob 0.15`
关闭：`--no-wta`
代码：[models/network.py](../models/network.py) `create_wm_network` 中 `iwm_*` 部分

---

### 2.4 ④ Mem 池专属 DC 偏置 `mem_bg_offset`（kernel 层，Scheme-B）

只对 Mem-A∪Mem-B 的背景电流额外加一个负的常数：

```python
I_bg = bg_mean + bg_std·noise + mem_bg_offset · mem_pool_mask
```

效果：把 Mem 池静息工作点压到双稳态的 LOW 态（≈0–1 Hz 静默），cue 提供的 NMDA 饱和把它"翻"到 HIGH 态后，移除 cue 仍能保持。

| 取值 | 物理含义 |
|------|--------|
| 0 | 关闭，行为同原模型 BG |
| -10 ~ -15 pA | 当前 v10 / v11 工作点（ΔV_inf ≈ -1~-1.5 mV） |
| -25 ~ -35 pA | 严格 Scheme-B 下行偏置 |

CLI：`--mem-bg-offset -10`（默认 0）
代码：[models/kernels.py](../models/kernels.py) `run_wm_kernel_dual`，`I_bg = I_bg + mem_bg_offset * mem_pool_mask`

---

### 2.5 ⑤ 脉冲频率适应 SFA（kernel 层）

仅对 Mem 池神经元额外维护一个慢的超极化电流 `w_adapt`：

```
spike   →  w_adapt += sfa_b           (典型 0.8 pA)
每步    →  w_adapt *= exp(-dt/sfa_tau)  (τ ≈ 2000 ms)
V 更新  →  I_total -= w_adapt
```

物理意义：模拟 Ca²⁺ 依赖的 K⁺ 通道 (I_AHP)，让 delay 期出现自然 ~20–30% 的缓慢下行，更接近真实 PFC 数据，而不是死板的水平线。

CLI：`--sfa-b 0.8 --sfa-tau 2000`
关闭：`--sfa-b 0`
代码：[models/kernels.py](../models/kernels.py) `run_wm_kernel_dual`，`w_adapt` 状态变量

---

### 2.6 ⑥ 协议层：baseline / cue / delay / probe（实验层）

原 resume 实验只是稳态 100 s + DA 阶跃。WM Demo 是结构化时间窗：

```
t=0       baseline 5 s     cue 1.5 s     delay 5 s     probe 0.5 s
├──────────────┼─────────────┼─────────────────┼──────────┤
                       ↑                ↑
                  注 300 pA 到        什么都不注
                  Mem-A 神经元         看 Mem-A 是否还高
```

DA 全程钉在 2 nM（与 ckpt 匹配）；`run_wm_kernel_dual` 同时支持 cue-A、cue-B 双通道，留作后续 distractor 实验。

CLI：`--baseline-ms / --cue-ms / --delay-ms / --probe-ms / --cue-amp`
代码：[simulation/runners.py](../simulation/runners.py) `run_wm_simulation_from_checkpoint`

---

## 三、改动清单速查表

| 模块 | 文件 | 改了什么 |
|------|------|---------|
| 网络结构 | `models/network.py :: create_wm_network` | 切 Mem-A/B 池、池内密化、切 I-WM、Mem↔I-WM 双向连接、把 W 拆成 `W_ampa_t` + `W_nmda_t` |
| 内核 | `models/kernels.py :: run_wm_kernel_dual` | 双通道前向 (AMPA+NMDA)、NMDA 饱和门控、`mem_bg_offset`、SFA `w_adapt` |
| Runner | `simulation/runners.py :: run_wm_simulation_from_checkpoint` | 从 ckpt 恢复、构建 cue-A/B mask、协议时间窗、记录 `wm_protocol` 元信息 |
| 实验 | `experiments/exp_wm_demo.py` | CLI 包装、参数透传、调用 `plot_wm_overview` 等 WM 专属可视化 |
| 可视化 | `analysis/wm_plotting.py` | `plot_wm_overview` / `plot_wm_rates_pools` / `compute_wm_metrics` / `print_wm_report` |
| 配置 | `config.py` | 新增 `WM_*` 全部默认值、`TAU_AMPA/TAU_NMDA/NMDA_ALPHA_GATE`、kernel params 表扩到 28 项 |

---

## 四、关键参数当前默认值（`config.py`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `WM_POOL_SIZE` | 100 | 每个 Mem 池神经元数 |
| `WM_INTRA_PROB` | 0.5 | Mem 池内 E→E 连接概率 |
| `WM_INTRA_W` | 4.0 pA | Mem 池内 E→E 权重（NMDA 通道） |
| `WM_CROSS_PROB` | 0.0 | Mem-A ↔ Mem-B 跨池连接 |
| `WM_USE_WTA` | True | 是否启用 WTA 共享抑制 |
| `WM_IWM_SIZE` | 60 | I-WM 子池神经元数（占 N_I 的 ~34%） |
| `WM_E2I_PROB / W` | 0.20 / +3.0 pA | Mem → I-WM |
| `WM_I2E_PROB / W` | 0.15 / -3.5 pA | I-WM → Mem |
| `WM_BASELINE_MS` | 5000 ms | 基线窗 |
| `WM_CUE_DURATION_MS` | 1500 ms | cue 持续 |
| `WM_DELAY_MS` | 5000 ms | delay 窗 |
| `WM_PROBE_MS` | 500 ms | probe 窗 |
| `WM_CUE_AMPLITUDE` | 350 pA | cue 注入电流 |
| `WM_MEM_BG_OFFSET` | 0.0 pA | Mem 池专属 DC（Scheme-B 关键钮） |
| `WM_SFA_B` | 0.8 pA | SFA 增量 |
| `WM_SFA_TAU` | 2000 ms | SFA 衰减 τ |
| `TAU_AMPA / TAU_NMDA` | 5 / 100 ms | 双通道时间常数 |
| `NMDA_ALPHA_GATE` | 0.5 | NMDA 饱和增益（v10/v11 调到 0.15） |

---

## 五、判断"是否真的实现了工作记忆"的三条硬指标

1. **基线 (0–5 s)**：Mem-A ≈ Mem-B ≈ 0–2 Hz（双稳态 LOW）
2. **cue (5–6.5 s)**：仅 Mem-A 被点亮到 ~30–50 Hz；Mem-B 因 WTA 仍 ≈ 0 Hz
3. **delay (6.5–11.5 s, cue 已撤销)**：
   - **Mem-A 仍维持高发放**（HIGH 吸引子，平台或缓慢下行均可）
   - **Mem-B 始终被压住**（选择性）
   - **E-BG / D1 / D2 / I 池基线不被卷入**

`wm_report.txt` 已自动给出每个窗口的均值并打印 HIGH 状态判定 (`Δ(Late−BL)`)。

