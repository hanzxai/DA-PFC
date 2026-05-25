# Working-Memory 模型公式集（case: `A_iwm_fixed_DA2to6_freshckpt`）

> **对应实验**：`outputs/exp_2026-05-12_09-55-31_A_iwm_fixed_DA2to6_freshckpt_DA2to6nM`
>
> **所用 kernel**：单通道线性突触版（`run_wm_kernel`，**非** `run_wm_kernel_dual`）；那时尚未把 NMDA 拆成饱和门控变量。
>
> **Scheme**：A — 结构化池（Mem-A、Mem-B、E-BG、I-D1、I-D2、I-WM）+ 显式 WTA 抑制群。
>
> **结论**：Δ(Late-BL) Mem-A = +12.62 Hz、Mem-B = +0.00 Hz，selectivity 完美；persistence ratio 23.5%。

---

## 1. 神经元（LIF）

### 1.1 膜电位

$$
C_m\,\frac{dV_i}{dt} \;=\; -\frac{V_i - V_{rest}}{R_{eff,i}(t)} \;+\; I_{total,i}(t)
$$

### 1.2 发放与复位

$$
V_i(t^-) \ge V_{th} \;\Longrightarrow\; \text{spike}_i(t)=1,\;\; V_i(t^+) \leftarrow V_{reset},\;\; V \text{ 在 } t_{ref} \text{ 内钳位}
$$

### 1.3 离散更新（Exponential Euler）

令 $V_\infty = V_{rest} + R_{eff}\,I_{total}$，$\tau_m = R_{eff}\,C_m$：

$$
V_i(t+\Delta t) \;=\; V_\infty + \big(V_i(t) - V_\infty\big)\,e^{-\Delta t / \tau_m}
$$

---

## 2. 总输入电流

$$
\boxed{\;I_{total,i}(t) \;=\; I_{syn,i}(t) \;+\; I_{bg,i}(t) \;+\; I_{cue,i}(t) \;+\; I_{DA,i}(t)\;}
$$

> 这里**没有**双通道 scale 因子、**没有** SFA 项（本 case 设 $b_{sfa}=0$）。

---

## 3. 突触动力学（**单通道，线性 EPSC**）

$$
\boxed{\;\tau_{syn}\,\frac{dI_{syn,i}}{dt} \;=\; -I_{syn,i} \;+\; \tau_{syn}\sum_j W_{ij}\,\sum_k \delta\!\left(t - t_j^{(k)}\right)\;}
$$

离散等价（先衰减后跳变）：

$$
I_{syn,i}(t+\Delta t) \;=\; I_{syn,i}(t)\,e^{-\Delta t / \tau_{syn}} \;+\; \sum_j W_{ij}\,\text{spike}_j(t)
$$

> 单一时间常数 $\tau_{syn}$，**没有 NMDA 饱和**（$s_j$ 不存在）；attractor 主要靠 recurrent 权重 $W_{ij}$ 的强度 + 网络结构维持。

### 3.1 连接权重（结构化池）

记 $J_{intra}=5.5$ pA、$p_{intra}=0.6$、$J_{E\to I_{wm}}=3.0$ pA、$p_{E\to I_{wm}}=0.2$、$J_{I_{wm}\to E}=-3.5$ pA、$p_{I_{wm}\to E}=0.15$：

$$
W_{ij} \;=\;
\begin{cases}
J_{intra} & i,j \in \mathrm{Mem\text{-}A},\;\text{w.p. } p_{intra} \\[2pt]
J_{intra} & i,j \in \mathrm{Mem\text{-}B},\;\text{w.p. } p_{intra} \\[2pt]
J_{E\to I_{wm}} & j \in \mathrm{Mem\text{-}A}\cup\mathrm{Mem\text{-}B},\; i \in \mathrm{I\text{-}WM},\;\text{w.p. } p_{E\to I_{wm}} \\[2pt]
J_{I_{wm}\to E} & j \in \mathrm{I\text{-}WM},\; i \in \mathrm{Mem\text{-}A}\cup\mathrm{Mem\text{-}B},\;\text{w.p. } p_{I_{wm}\to E} \\[2pt]
W^{ckpt}_{ij} & \text{otherwise (来自 ckpt\_DA2nM\_bg200\_100s)}
\end{cases}
$$

---

## 4. 背景电流

$$
I_{bg,i}(t) \;=\; \mu_{bg} \;+\; \sigma_{bg}\,\xi_i(t) \;+\; \Delta I_{mem}\,\mathbb{1}\!\left[i \in \mathrm{Mem\text{-}A} \cup \mathrm{Mem\text{-}B}\right]
$$

- $\mu_{bg} = 200$ pA、$\xi_i(t) \sim \mathcal{N}(0,1)$ 独立白噪声
- $\Delta I_{mem} = -18.0$ pA（**Scheme-A 关键负偏置**：把 Mem 池的静息发放压到 ≈0 Hz，让 WTA / cue / 自激能干净地把它从 down-state 拉到 up-state）
- Batch-0（Control）和 Batch-1（Exp）共享同一份 $\xi$ 抽样（pair-matched）

---

## 5. Cue 注入

只在 cue 窗口 $[t_{cue\_on}, t_{cue\_off}) = [5000, 6500)$ ms、只对 Mem-A：

$$
I_{cue,i}(t) \;=\; A_{cue}\,\mathbb{1}\!\big[t \in [t_{cue\_on}, t_{cue\_off})\big]\,M^{A}_i, \qquad A_{cue} = 300\,\mathrm{pA}
$$

$M^{A}_i \in \{0,1\}$ 为 Mem-A 成员掩码，$|\mathrm{Mem\text{-}A}|=60$。

---

## 6. DA 浓度时序（梯形）

Batch-0 Control：

$$
\mathrm{DA}^{(0)}(t) \;\equiv\; 2.0\,\mathrm{nM}
$$

Batch-1 Exp（窗口 $[t_{on}, t_{off}) = [5000, 11500)$ ms，覆盖 cue+delay）：

$$
\mathrm{DA}^{(1)}(t) \;=\;
\begin{cases}
2.0 & t < t_{on}\;\text{or}\;t \ge t_{off} \\[2pt]
2.0 + 4.0 \cdot \dfrac{t - t_{on}}{T_{up}} & t_{on} \le t < t_{on}+T_{up} \\[8pt]
6.0 & t_{on}+T_{up} \le t < t_{off}-T_{down} \\[2pt]
2.0 + 4.0 \cdot \dfrac{t_{off} - t}{T_{down}} & t_{off}-T_{down} \le t < t_{off}
\end{cases}
$$

（单位：nM；$T_{up}, T_{down}$ 为可配的上升/下降斜坡时长。）

---

## 7. DA 受体动力学

### 7.1 Sigmoid 浓度响应（瞬时靶值）

$$
S_X\!\big(\mathrm{DA}(t)\big) \;=\; \frac{1}{1 + \exp\!\big(-\beta\,(\mathrm{DA}(t) - \mathrm{EC50}_X)\big)}, \qquad X \in \{D1, D2\}
$$

### 7.2 Langmuir 占用率 ODE

$$
\frac{d\alpha_X}{dt} \;=\; k^X_{on}\,S_X(\mathrm{DA})\,(1 - \alpha_X) \;-\; k^X_{off}\,\alpha_X
$$

稳态：

$$
\alpha_{X,\infty} \;=\; \frac{k^X_{on}\,S_X(\mathrm{DA})}{k^X_{on}\,S_X(\mathrm{DA}) + k^X_{off}}
$$

### 7.3 受体掩码

$$
m_{D1,i} = \mathbb{1}[i \in \mathrm{D1\text{ segment}}], \qquad
m_{D2,i} = \mathbb{1}[i \in \mathrm{D2\text{ segment}}]
$$

> 本 case `block_d1=False, block_d2=False`，两路 DA 都开。

---

## 8. DA 对电路的三条作用路径

### 8.1 Gain（膜电阻）

$$
R_{eff,i}(t) \;=\; R_{base}\,\Big(1 \;+\; \varepsilon_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;-\; \varepsilon_{D2}\,\alpha_{D2}(t)\,m_{D2,i}\Big)
$$

### 8.2 Bias（DC 注入）

$$
I_{DA,i}(t) \;=\; \mathrm{BIAS}_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;+\; \mathrm{BIAS}_{D2}\,\alpha_{D2}(t)\,m_{D2,i}
$$

### 8.3 Synaptic Scaling（仅作用于 cue / 部分突触；本 case 该项已并入 $I_{syn}$ 等价表达）

$$
I_{syn,i}^{(\mathrm{eff})}(t) \;=\; \big(1 + \lambda_{D1}\,\alpha_{D1}(t)\,m_{D1,i} - \lambda_{D2}\,\alpha_{D2}(t)\,m_{D2,i}\big)\,I_{syn,i}(t)
$$

---

## 9. WTA（I-WM 抑制群）平衡

I-WM 群（$N_{IWM}=60$）的稳态发放率近似：

$$
r_{IWM} \;\approx\; \Phi\!\Big(V_{rest} + R_{eff,IWM}\big[\,p_{E\to I_{wm}}\,J_{E\to I_{wm}}\,N_{Mem}\,r_{Mem} \;+\; \mu_{bg}\,\big]\Big)
$$

其反馈到 Mem-A 的总抑制电流：

$$
I_{wta\to A}(t) \;=\; p_{I_{wm}\to E}\,J_{I_{wm}\to E}\,N_{IWM}\,\bar s_{IWM}(t) \quad (J_{I_{wm}\to E}<0)
$$

> $\bar s_{IWM}$ 这里就是 I-WM 群在 $\tau_{syn}$ 时间常数下的平均突触状态（单通道，无 NMDA 饱和）。

---

## 10. Mem-A 池的 mean-field 自洽方程

设 Mem-A 稳态发放率 $r_A$，**delay 相**（cue 关、probe 前）：

### 10.1 池内 recurrent 自激

单通道线性突触下，稳态有 $\langle I_{syn} \rangle \propto \tau_{syn}\,r$，故：

$$
I_{rec}(r_A) \;=\; \tau_{syn}\,N_{pool}\,p_{intra}\,J_{intra}\,r_A
$$

（$N_{pool}=60$, $p_{intra}=0.6$, $J_{intra}=5.5$ pA）

### 10.2 WTA 反馈

$$
I_{wta}(r_A) \;=\; \tau_{syn}\,p_{I_{wm}\to E}\,J_{I_{wm}\to E}\,N_{IWM}\,r_{IWM}(r_A)
$$

### 10.3 V_∞ 总驱动

$$
V_{\infty,A}(r_A) \;=\; V_{rest} + R_{eff}(\alpha_{D1},\alpha_{D2})\,\Big[\,I_{rec}(r_A) + I_{wta}(r_A) + \mu_{bg} + \Delta I_{mem} + \mathrm{BIAS}_{D1}\,\alpha_{D1}\,\Big]
$$

### 10.4 Siegert 自洽

$$
r_A \;=\; \Phi\!\big(V_{\infty,A}(r_A);\, V_{th}, V_{reset}, \sigma_V, \tau_m, t_{ref}\big)
$$

> **这里没有 $s_\infty(r)$ 饱和项**（单通道线性），所以 attractor 完全由 $J_{intra}\,N_{pool}\,p_{intra}\,\tau_{syn}$ 与 WTA 之间的非线性平衡（来自 $\Phi$ 的阈值非线性）决定。这也是该 case **persistence ratio 只有 23.5% / late phase 仍在缓慢衰减**的根本原因 —— 没有 NMDA 饱和锁住 up-state。

---

## 11. 状态变量与 timestep 更新顺序

```
1. 计算 DA(t)              ← 第 6 节
2. 算 S_D1(DA), S_D2(DA)   ← 第 7.1 节
3. α_X 一阶 Euler 更新     ← 第 7.2 节
4. 算 R_eff, I_DA          ← 第 8 节
5. 加 cue 注入 I_cue       ← 第 5 节
6. 衰减 I_syn              ← 第 3 节衰减项
7. 算 I_bg                 ← 第 4 节
8. 装配 I_total            ← 第 2 节
9. 更新 V (Exp Euler)      ← 第 1.3 节
10. 检测 spike, V≥V_th
11. V ← V_reset, 进入 t_ref
12. spike-driven: I_syn += spike·W
```

> **本 case 不更新** $s_j$（无 NMDA 门控）、**不更新** $w_{adapt}$（$b_{sfa}=0$）。

---

## 12. 本 case 关键参数清单（来自 [config.json](../outputs/exp_2026-05-12_09-55-31_A_iwm_fixed_DA2to6_freshckpt_DA2to6nM/config.json)）

| 参数 | 值 | 公式中位置 |
|------|-----|-----------|
| `pool_size` $N_{pool}$ | 60 | §3.1, §10.1 |
| `intra_prob` $p_{intra}$ | 0.6 | §3.1, §10.1 |
| `intra_w` $J_{intra}$ | 5.5 pA | §3.1, §10.1 |
| `iwm_size` $N_{IWM}$ | 60 | §9 |
| `e2i_prob` $p_{E\to I_{wm}}$ | 0.2 | §3.1 |
| `e2i_w` $J_{E\to I_{wm}}$ | 3.0 pA | §3.1 |
| `i2e_prob` $p_{I_{wm}\to E}$ | 0.15 | §3.1, §9 |
| `i2e_w` $J_{I_{wm}\to E}$ | −3.5 pA | §3.1, §9 |
| `mem_bg_offset` $\Delta I_{mem}$ | −18.0 pA | §4 |
| `bg_mean` $\mu_{bg}$ | 200 pA | §4 |
| `cue_amplitude` $A_{cue}$ | 300 pA | §5 |
| `cue_ms` | 1500 ms | §5 |
| `baseline_ms / delay_ms / probe_ms` | 5000 / 5000 / 500 ms | §5, §6 |
| `da_base` $\mathrm{DA}_{base}$ | 2.0 nM | §6 |
| `da_pulse` $\mathrm{DA}_{peak}$ | 6.0 nM | §6 |
| `da_pulse_onset / offset` | 5000 / 11500 ms | §6 |
| `sfa_b` | 0.0 → 无 SFA | （§7 of MODEL_EQUATIONS 不启用） |
| `block_d1 / block_d2` | False / False | §8 |

---

## 13. 与 v2 双通道版的差异（一句话）

| 维度 | 本 case (单通道) | dual-channel 版（[MODEL_EQUATIONS.md](MODEL_EQUATIONS.md)） |
|------|------------------|---------------------------------------|
| 突触状态变量 | 后突触 $I_{syn,i}$ | 前突触 $s_j$ + 后突触 $I_{ampa,i}$ |
| 饱和性 | 线性 | NMDA $s_j \in [0,1]$ 饱和 |
| attractor 机制 | $\Phi(\cdot)$ 阈值非线性 + WTA | NMDA 饱和 + WTA |
| persistence | 缓慢衰减（23.5%） | 锁死可达 ~100% |

---

*此文档严格对应 case `A_iwm_fixed_DA2to6_freshckpt`；如需 dual-channel 通用公式见 [MODEL_EQUATIONS.md](MODEL_EQUATIONS.md)。*
