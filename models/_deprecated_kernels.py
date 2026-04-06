# models/_deprecated_kernels.py
"""
Archived / deprecated kernel helper functions.

These functions are no longer called by any active kernel but are preserved
here for reference and potential future use.

Archived on: 2026-04-06
Reason: Replaced by Langmuir versions (compute_alpha_d*_step_langmuir)
        and parameter-table approach (verify_kernel_params_consistent no longer needed).
"""
import torch


# ======================================================================
# [DEPRECATED] alpha_D1 tau version — replaced by Langmuir version
# ======================================================================
@torch.jit.script
def compute_alpha_d1_step(
    alpha_d1: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
) -> torch.Tensor:
    """
    alpha_D1 一阶动力学单步更新 (tau 版本)。
    已被 compute_alpha_d1_step_langmuir 替代。
    """
    TAU_ON_D1  = 30876.1
    TAU_OFF_D1 = 164472.5
    EC50_D1    = 4.0
    BETA       = 1.0

    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

    diff = s_d1 - alpha_d1
    tau_dynamic = torch.where(
        diff > 0,
        torch.tensor(TAU_ON_D1,  device=alpha_d1.device),
        torch.tensor(TAU_OFF_D1, device=alpha_d1.device),
    )
    alpha_d1_new = alpha_d1 + (diff / tau_dynamic) * dt
    return alpha_d1_new


# ======================================================================
# [DEPRECATED] alpha_D1 k_on/k_off version — replaced by Langmuir version
# ======================================================================
@torch.jit.script
def compute_alpha_d1_step_kon_koff(
    alpha_d1: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D1 一阶动力学单步更新 — k_on / k_off 参数化版本。
    已被 compute_alpha_d1_step_langmuir 替代。

    方程: dα/dt = k_on · (s_D1 - α)⁺ - k_off · (α - s_D1)⁺
    稳态: α_ss = s_D1
    """
    EC50_D1 = 4.0
    BETA    = 1.0

    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

    rise_term = torch.clamp(s_d1 - alpha_d1, min=0.0)
    fall_term = torch.clamp(alpha_d1 - s_d1, min=0.0)

    d_alpha = k_on * rise_term - k_off * fall_term
    alpha_d1_new = alpha_d1 + d_alpha * dt
    return alpha_d1_new


# ======================================================================
# [DEPRECATED] alpha_D2 tau version — replaced by Langmuir version
# ======================================================================
@torch.jit.script
def compute_alpha_d2_step(
    alpha_d2: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
) -> torch.Tensor:
    """
    alpha_D2 一阶动力学单步更新 (tau 版本)。
    已被 compute_alpha_d2_step_langmuir 替代。
    """
    TAU_ON_D2  = 10000.0
    TAU_OFF_D2 = 50000.0
    EC50_D2    = 8.0
    BETA       = 1.0

    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

    diff = s_d2 - alpha_d2
    tau_dynamic = torch.where(
        diff > 0,
        torch.tensor(TAU_ON_D2,  device=alpha_d2.device),
        torch.tensor(TAU_OFF_D2, device=alpha_d2.device),
    )
    alpha_d2_new = alpha_d2 + (diff / tau_dynamic) * dt
    return alpha_d2_new


# ======================================================================
# [DEPRECATED] alpha_D2 k_on/k_off version — replaced by Langmuir version
# ======================================================================
@torch.jit.script
def compute_alpha_d2_step_kon_koff(
    alpha_d2: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D2 一阶动力学单步更新 — k_on / k_off 参数化版本。
    已被 compute_alpha_d2_step_langmuir 替代。

    方程: dα/dt = k_on · (s_D2 - α)⁺ - k_off · (α - s_D2)⁺
    稳态: α_ss = s_D2
    """
    EC50_D2 = 8.0
    BETA    = 1.0

    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

    rise_term = torch.clamp(s_d2 - alpha_d2, min=0.0)
    fall_term = torch.clamp(alpha_d2 - s_d2, min=0.0)

    d_alpha = k_on * rise_term - k_off * fall_term
    alpha_d2_new = alpha_d2 + d_alpha * dt
    return alpha_d2_new


# ======================================================================
# [DEPRECATED] Parameter consistency check — no longer needed after
# parameter-table approach (params are passed in, not hardcoded)
# ======================================================================
def verify_kernel_params_consistent() -> None:
    """
    校验 kernels.py 中硬编码的参数与 config.py 是否一致。
    不一致时抛出 AssertionError, 并打印具体差异。

    已废弃: 参数表传参方案实施后, kernel 不再硬编码参数,
    此函数不再需要。
    """
    import sys
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import config as cfg  # type: ignore

    _KERNEL_PARAMS: dict = {
        "TAU_ON_D1":  30876.1,
        "TAU_OFF_D1": 164472.5,
        "TAU_ON_D2":  10000.0,
        "TAU_OFF_D2": 50000.0,
        "EC50_D1":    4.0,
        "EC50_D2":    8.0,
        "BETA":       1.0,
        "EPS_D1":     0.15,
        "EPS_D2":     0.10,
        "BIAS_D1":    12.0,
        "BIAS_D2":   -10.0,
        "LAM_D1":     0.3,
        "LAM_D2":     0.2,
        "V_REST":     -70.0,
        "V_RESET":   -75.0,
        "V_TH":       -50.0,
        "R_BASE":     0.1,
        "C_E":        250.0,
        "C_I":        90.0,
        "TAU_SYN":    5.0,
        "T_REF":      5.0,
        "BG_MEAN":    200.0,
        "BG_STD":     25.0,
        "DA_BASELINE": 2.0,
    }

    errors: list = []
    for name, kernel_val in _KERNEL_PARAMS.items():
        cfg_val = getattr(cfg, name, None)
        if cfg_val is None:
            errors.append(f"  [MISSING] config.{name} not found")
            continue
        if abs(float(cfg_val) - float(kernel_val)) > 1e-6:
            errors.append(
                f"  [MISMATCH] {name}: config={cfg_val}, kernel={kernel_val}"
            )

    if errors:
        msg = (
            "\n\n[kernels.py] Parameter consistency check FAILED!\n"
            "The following parameters differ between config.py and kernels.py.\n"
            "Please update kernels.py to match config.py (or vice versa):\n\n"
            + "\n".join(errors)
            + "\n\nHint: Search for the parameter name in kernels.py and update the literal value.\n"
        )
        raise AssertionError(msg)

    print("[kernels.py] Parameter consistency check PASSED.")
