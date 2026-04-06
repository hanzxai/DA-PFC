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
TAU_SYN = 5.0       # 突触时间常数 (ms)
T_REF = 5.0         # 不应期 (ms)

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
EPS_D1 = 0.15        # D1: Gain 增强比例 (alpha=1 时 R_eff = R_base * 1.15)
EPS_D2 = 0.10        # D2: Gain 减弱比例 (alpha=1 时 R_eff = R_base * 0.90)

BIAS_D1 = 12.0       # D1: 偏置电流 (pA) [DA=15nM时: 12*0.65≈7.8pA, ΔV≈+2.8mV]
BIAS_D2 = -10.0      # D2: 偏置电流 (pA)

LAM_D1 = 0.3         # D1: 突触缩放 (Synaptic Scaling)
LAM_D2 = 0.2         # D2: 突触缩放 (Synaptic Scaling)

# ==============================================================================
# 8. Spike 记录参数
# ==============================================================================
SPIKE_RATE_ESTIMATE = 0.15  # 预估最大发放占比 (用于分配记录缓冲区)


# ==============================================================================
# 9. Kernel 参数表 (Parameter Table for JIT kernels)
#    将所有 kernel 需要的参数打包成一个 1-D Tensor, 作为函数参数传入。
#    这样 @torch.jit.script 函数不再需要硬编码参数, 修改只需改 config.py。
#
#    Index layout (25 elements):
#      [0]  V_REST       [1]  V_RESET      [2]  V_TH
#      [3]  R_BASE       [4]  TAU_SYN      [5]  T_REF
#      [6]  BG_MEAN      [7]  BG_STD       [8]  C_E
#      [9]  C_I          [10] EC50_D1      [11] EC50_D2
#      [12] BETA         [13] EPS_D1       [14] EPS_D2
#      [15] BIAS_D1      [16] BIAS_D2      [17] LAM_D1
#      [18] LAM_D2       [19] TAU_ON_D1    [20] TAU_OFF_D1
#      [21] TAU_ON_D2    [22] TAU_OFF_D2   [23] DA_BASELINE
#      [24] SPIKE_RATE_ESTIMATE
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
    ], dtype=torch.float64, device=device)
