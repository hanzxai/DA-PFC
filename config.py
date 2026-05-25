# config.py
"""
DA-PFC 项目全局配置
所有生理学、药理学参数的唯一定义位置 (Single Source of Truth)
"""
import torch

# ==============================================================================
# 1. 硬件配置 (Hardware)
# ==============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. 网络结构参数 (Network Architecture)
# ==============================================================================
N_E = 824           # 兴奋性神经元数量 (82.4%)
N_I = 176           # 抑制性神经元数量 (17.6%)
N_TOTAL = N_E + N_I

# 连接概率与权重
# [对齐 Gemini 版本] 稀疏连接 + 弱权重, 等效递归兴奋: 0.02*800*3.0*0.1 ≈ 4.8
CONN_PROB = 0.02    # 连接概率 (Gemini: 0.02)
W_EXC = 3.0         # 兴奋性权重 (pA) [Gemini等效: 0.3/R_base=3.0 pA]
W_INH = -20.0       # 抑制性权重 (pA) [Gemini等效: -2.0/R_base=-20.0 pA]

# 受体表达比例
FRAC_E_D1 = 0.21    # E 神经元中 D1R 比例 (21% of 824 = 173)
FRAC_E_D2 = 0.25    # E 神经元中 D2R 比例 (25% of 824 = 206)
FRAC_I_D1 = 0.30    # I 神经元中 D1R 比例 (30% of 176 = 52)
FRAC_I_D2 = 0.08    # I 神经元中 D2R 比例 (8%  of 176 = 14)

# ==============================================================================
# 3. LIF 神经元参数 (Leaky Integrate-and-Fire)
# ==============================================================================
V_REST = -70.0      # 静息电位 (mV)  [生理值: -70 mV]
V_RESET = -75.0     # 复位电位 (mV)  [生理值: 超极化复位 -75 mV]
V_TH = -50.0        # 阈值电位 (mV)  [生理值: -50 mV]
R_BASE = 0.1        # 基础膜电阻 (GΩ) = 100 MΩ  [单位: mV/pA = GΩ]
C_E = 250.0         # 兴奋性神经元膜电容 (pF)  [τ_m(E) = R_BASE * C_E = 25 ms]
C_I = 90.0          # 抑制性神经元膜电容 (pF)  [τ_m(I) = R_BASE * C_I = 9 ms]
TAU_SYN = 5.0       # 突触时间常数 (ms) [Legacy single-channel; AMPA-fast]
T_REF = 5.0         # 不应期 (ms)

# ── 双通道突触 (WM kernel) ──
# 普通 E→E、E→I、I→E 通道走 AMPA/GABA 快突触，记忆池 intra E→E 走 NMDA 慢突触。
# AMPA = 5 ms 让 I-WM 反馈能即时响应（不被 100ms 拖累 → WTA 真正生效）；
# NMDA = 100 ms 提供池内积分器，撑起 attractor。
# 仅 run_wm_kernel_dual 使用，不影响其他 kernel。
TAU_AMPA = 5.0      # 快通道突触时间常数 (ms) [AMPA + GABA-A]
TAU_NMDA = 100.0    # 慢通道突触时间常数 (ms) [NMDA, Wang 2002]

# NMDA saturation gain (Wang 2002):
#   ds_j/dt = -s_j/tau_NMDA + alpha_gate * (1 - s_j) * spike_j
# In Wang 2002 alpha = 0.5 (per spike s jumps half-way to saturation).
# This is the *core* nonlinearity that gives the network a stable high-rate
# fixed point — without it, recurrent EPSC grows linearly and the network
# either runs away or decays passively after cue offset.
NMDA_ALPHA_GATE    = 0.5

# 背景输入
# V_ss = V_rest + R_base * I_bg = -70 + 0.1 * 200 = -50 mV = V_th
# 临界点驱动: V_inf ≈ V_th, 发放由噪声波动驱动 (fluctuation-driven regime)
# 噪声: BG_STD=25 pA → σ_V = R_base * BG_STD = 0.1 * 25 = 2.5 mV
# 理论发放率 (Siegert): E ~8-10 Hz, I ~25-30 Hz
# DA BIAS 调制: 20 pA × α ≈ 13 pA → ΔV_inf ≈ 1.3 mV ≈ 0.5σ → ΔRate >15%
BG_MEAN = 200.0     # 背景电流均值 (pA)  [V_ss=-50mV, 临界点驱动]
BG_STD = 25.0       # 背景电流标准差 (pA)  [σ_V=2.5mV, 精细噪声控制]

# ==============================================================================
# 4. 仿真参数 (Simulation)
# ==============================================================================
DT = 1.0            # 时间步长 (ms) [暂时用1.0快速测试, 后续可改0.1提高精度]
DEFAULT_DURATION = 15000.0  # 默认仿真时长 (ms)
DEFAULT_DA_ONSET = 10000.0  # 默认给药时间 (ms)  [必须 < DEFAULT_DURATION]
DA_BASELINE = 2.0           # 基线 DA 浓度 (nM)  [Control 和 Exp 的 baseline 均为 2nM]
RANDOM_SEED = 42             # 随机种子

# ==============================================================================
# 5. 受体动力学参数 (Receptor Kinetics)
#    数据来源：论文
# ==============================================================================

# D1 受体时间常数 (ms)
TAU_ON_D1 = 30876.1      # 上升阶段: if S > alpha
TAU_OFF_D1 = 164472.5    # 衰减阶段: if S <= alpha

# D2 受体时间常数 (ms) — D2 动力学比 D1 快约 3x (偶联 Gi 蛋白, 信号链更短)
TAU_ON_D2 = 10000.0       # 上升阶段 (ms)
TAU_OFF_D2 = 50000.0      # 衰减阶段 (ms)

# ==============================================================================
# 6. 药理学参数 (Pharmacology)
#    Sigmoid 激活函数: S = 1 / (1 + exp(-BETA * (DA - EC50)))
# ==============================================================================
BETA = 1.0           # Sigmoid 斜率
EC50_D1 = 4.0        # D1 半效浓度 (nM)
EC50_D2 = 8.0        # D2 半效浓度 (nM)

# ==============================================================================
# 7. 调节强度参数 (Modulation Strength)
#    alpha=1.0 时对膜参数的影响幅度
# ==============================================================================
EPS_D1 = 0.015       # D1: Gain 增强比例 (alpha=1 时 R_eff = R_base * 1.015)
EPS_D2 = 0.01        # D2: Gain 减弱比例 (alpha=1 时 R_eff = R_base * 0.99)

BIAS_D1 = 3.0        # D1: 偏置电流 (pA)
BIAS_D2 = -3.0       # D2: 偏置电流 (pA)

LAM_D1 = 0.3         # D1: 突触缩放 (Synaptic Scaling)
LAM_D2 = 0.2         # D2: 突触缩放 (Synaptic Scaling)

# ==============================================================================
# 8. Spike 记录参数
# ==============================================================================
SPIKE_RATE_ESTIMATE = 0.15  # 预估最大发放占比 (用于分配记录缓冲区)

# ==============================================================================
# 8b. Working-Memory pool parameters (Scheme A: structured selective sub-pools)
#     Carve two memory pools out of E-Other (445 neurons available).
#     Pool A receives the cue; Pool B serves as a competing distractor pool.
#     Strong intra-pool recurrence + zero cross-pool connectivity gives
#     winner-take-all persistent activity within each pool.
#
#     IMPORTANT — these defaults are tuned for the LOW-BG (sub-threshold)
#     regime (BG_MEAN ≈ 160 pA, V_inf ≈ -54 mV). Under the standard
#     critical-point BG (200 pA, V_inf = V_th), the recurrent pool becomes
#     bistable and immediately latches to ~120 Hz at t=0, masking any cue.
#     The recommended workflow is therefore:
#       1) generate a baseline ckpt with --bg-mean 160
#       2) run exp_wm_demo.py with --bg-mean 160 on that ckpt
#     See scripts/run_wm_demo_lowbg.sh for the one-click pipeline.
# ==============================================================================
WM_POOL_SIZE       = 60     # neurons per memory pool (A, B)
                            # Mem-A,B both carved from E-D1 segment [0, 173):
                            #   Mem-A=[0,60), Mem-B=[60,120), D1-BG=[120,173).
                            # 60 < 100 keeps 2*pool+margin <= n_e_d1=173;
                            # 60-cell attractors are well within the Wang-2002
                            # operating range (16-128 in the original paper).
WM_NUM_POOLS       = 2      # only pool A & B used in the demo
# Self-recurrence (intra-pool):
#   Net gain G = N_pool * p * W * tau_syn(s) ≈ 100*0.25*3.5*0.1 = 8.75 pA·s
#   Crossed the bistable threshold: cue pushes Mem-A to ~55 Hz, then the
#   self-recurrent EPSC alone (≈ 8.75 * r ≈ 480 pA at r=55) is enough to
#   keep the pool above V_th even after the cue is removed.  The shared
#   I-WM loop is what suppresses Mem-B (selectivity), not what stabilises
#   Mem-A (that's pure self-recurrence).
# With Wang-2002 saturating NMDA, the recurrent EPSC at fixed point r is
#   I_NMDA(r) = N * p * J * s_inf(r),   s_inf = alpha*tau*r / (1 + alpha*tau*r)
# Plug in alpha=0.5, tau=0.1s, r=20Hz: s_inf ≈ 0.5 → I_NMDA ≈ 50*p*J pA.
# To get ΔV ≈ +10 mV at the high state (override the ~5 mV WTA inhibition
# plus 4 mV margin), need I_NMDA ≈ 100 pA -> N*p*J ≈ 200 pA.
# With N=100, p=0.5, J=4 -> 200 pA. ✓
# IMPORTANT: With saturation, J*p doesn't have to be huge — the gain comes
# from the s ramping up to ~1, not from linear accumulation.
WM_INTRA_PROB      = 0.5    # intra-pool connection probability (Wang attractor)
WM_INTRA_W         = 4.0    # intra-pool synaptic weight (pA, NMDA channel)
WM_CROSS_PROB      = 0.0    # A <-> B cross connectivity (kept 0 for competition)

# ----------------------------------------------------------------------
#  WTA (winner-take-all) shared inhibition.
#  Carve a dedicated "I-WM" sub-pool out of the existing inhibitory
#  population N_I, then strengthen the loop:
#       Mem-A, Mem-B  ---excitatory--->  I-WM  ---inhibitory--->  Mem-A, Mem-B
#  The shared inhibition implements winner-take-all: when Mem-A is high,
#  it strongly drives I-WM, which in turn suppresses BOTH pools; only the
#  pool with strong intrinsic recurrence (the cued one) can survive,
#  while the un-cued pool is pushed back to baseline.
#
#  Default values are deliberately conservative — the shared I sub-pool
#  is only 40 neurons (~23% of N_I) so the global inhibitory tone is
#  preserved.  Set WM_USE_WTA=False to disable the WTA structure entirely.
# ----------------------------------------------------------------------
WM_USE_WTA         = True   # enable shared-inhibition WTA structure
WM_IWM_SIZE        = 60     # neurons in I-WM sub-pool (carved from N_I=176, ~34%)
# ── Critical correction (post-mortem of the "Mem=0Hz forever" run) ──
# I-WM neurons receive the *normal* background drive (BG=200), so they fire
# at ~13 Hz at baseline regardless of Mem activity.  If we make the I-WM ->
# Mem projection too strong, the BASELINE inhibition alone clamps Mem in
# deep hyperpolarization (V << V_reset) and the cue can never recruit it.
#
# Budget at baseline (Mem ≈ 0 Hz, I-WM ≈ 13 Hz, target |ΔV_inhib| ≈ 4 mV):
#   I_inhib = N_iwm * p * |W| * tau_syn(s) * r_iwm
#           = 60   * 0.15 * 3.5 * 0.1 * 13 ≈ 41 pA  ->  ΔV = -4.1 mV  ✓
# Once Mem-A wins (~30 Hz), I-WM rises to ~22 Hz, lifting the inhibition on
# Mem-B to ~70 pA (ΔV = -7 mV), which together with Mem-B's lack of self-
# recurrent drive keeps Mem-B suppressed.  This is the WTA mechanism.
WM_E2I_PROB        = 0.20   # Mem-{A,B} -> I-WM connection probability
WM_E2I_W           = 3.0    # Mem-{A,B} -> I-WM synaptic weight (pA)
WM_I2E_PROB        = 0.15   # I-WM -> Mem-{A,B} connection probability
WM_I2E_W           = -3.5   # I-WM -> Mem-{A,B} synaptic weight (pA, INH)

# Default WM task protocol (ms, relative to t=0 of the WM run)
WM_BASELINE_MS     = 5000.0
WM_CUE_DURATION_MS = 1500.0
WM_DELAY_MS        = 5000.0
WM_PROBE_MS        = 500.0
WM_CUE_AMPLITUDE   = 350.0  # external current injected into Mem-A during cue (pA)

# ── Scheme-B: per-pool baseline-bias offset ──
# Add this constant DC offset (pA) ONLY to Mem-A∪Mem-B neurons.
# A NEGATIVE value (e.g. -30 pA → ΔV_inf = -3 mV) puts the memory pool in
# the LOW state of the bistable f-I curve at baseline (silent, ~0-1 Hz),
# while leaving E-BG / I / D1 / D2 neurons at their normal BG_MEAN drive.
# Cue-driven NMDA saturation provides the extra +5 mV needed to flip the
# pool into the HIGH attractor state, which then persists through delay
# even after the cue is removed.
# Default 0.0 keeps the old (BG-only) behaviour; set to -25..-35 pA for
# Scheme-B WM attractor at BG_MEAN=200.
WM_MEM_BG_OFFSET   = 0.0

# ── Spike-Frequency Adaptation (SFA) for Mem pools ──
# Biologically, cortical pyramidal neurons have Ca²⁺-dependent K⁺ channels
# (I_AHP) that produce a slow afterhyperpolarisation after sustained firing.
# This causes the firing rate to gradually decline over seconds — exactly
# the "natural decay" seen in real PFC delay-period activity.
#
# Implementation (Compte et al. 2000, Brette & Gerstner 2005):
#   Each Mem-pool neuron has an adaptation variable w (pA):
#     dw/dt = -w / tau_sfa           (passive decay)
#     w → w + b_sfa   on each spike  (Ca²⁺ influx → K⁺ channel opening)
#   w is subtracted from I_total (hyperpolarising current).
#
# Effect on WM:
#   - During cue (high r): w ramps up → rate starts declining
#   - During delay: w slowly decays (τ=1500ms) → rate recovers partially
#   - Net result: delay-period rate shows a slow ~20-30% decline over 5s
#     instead of a perfectly flat plateau. Much more biologically realistic.
#
# Set SFA_B=0 to disable adaptation entirely (reverts to flat attractor).
WM_SFA_B           = 0.8    # adaptation increment per spike (pA)
WM_SFA_TAU         = 2000.0 # adaptation decay time constant (ms)


# ==============================================================================
# 9. Kernel 参数表# 9. Kernel 参数表 (Parameter Table for JIT kernels)
#    将所有 kernel 需要的参数打包成一个 1-D Tensor, 作为函数参数传入。
#    这样 @torch.jit.script 函数不再需要硬编码参数, 修改只需改 config.py。
#
#    Index layout (27 elements):
#      [0]  V_REST       [1]  V_RESET      [2]  V_TH
#      [3]  R_BASE       [4]  TAU_SYN      [5]  T_REF
#      [6]  BG_MEAN      [7]  BG_STD       [8]  C_E
#      [9]  C_I          [10] EC50_D1      [11] EC50_D2
#      [12] BETA         [13] EPS_D1       [14] EPS_D2
#      [15] BIAS_D1      [16] BIAS_D2      [17] LAM_D1
#      [18] LAM_D2       [19] TAU_ON_D1    [20] TAU_OFF_D1
#      [21] TAU_ON_D2    [22] TAU_OFF_D2   [23] DA_BASELINE
#      [24] SPIKE_RATE_ESTIMATE             [25] TAU_AMPA  [26] TAU_NMDA
#      [27] NMDA_ALPHA_GATE (Wang-2002 NMDA saturation)
# ==============================================================================

def build_kernel_params(device: torch.device = None) -> torch.Tensor:
    """
    Build a 1-D parameter tensor for JIT kernel functions.

    All kernel-relevant parameters are packed into a single tensor so that
    @torch.jit.script functions can receive them as an argument instead of
    hardcoding values. This eliminates parameter duplication and ensures
    config.py is the Single Source of Truth.

    Args:
        device: Target device. Defaults to config.DEVICE.

    Returns:
        params: (25,) float tensor with the layout documented above.
    """
    if device is None:
        device = DEVICE
    return torch.tensor([
        V_REST,             # [0]
        V_RESET,            # [1]
        V_TH,               # [2]
        R_BASE,             # [3]
        TAU_SYN,            # [4]
        T_REF,              # [5]
        BG_MEAN,            # [6]
        BG_STD,             # [7]
        C_E,                # [8]
        C_I,                # [9]
        EC50_D1,            # [10]
        EC50_D2,            # [11]
        BETA,               # [12]
        EPS_D1,             # [13]
        EPS_D2,             # [14]
        BIAS_D1,            # [15]
        BIAS_D2,            # [16]
        LAM_D1,             # [17]
        LAM_D2,             # [18]
        TAU_ON_D1,          # [19]
        TAU_OFF_D1,         # [20]
        TAU_ON_D2,          # [21]
        TAU_OFF_D2,         # [22]
        DA_BASELINE,        # [23]
        SPIKE_RATE_ESTIMATE,  # [24]
        TAU_AMPA,           # [25] dual-channel only
        TAU_NMDA,           # [26] dual-channel only
        NMDA_ALPHA_GATE,    # [27] dual-channel only (NMDA saturation gain)
    ], dtype=torch.float64, device=device)
