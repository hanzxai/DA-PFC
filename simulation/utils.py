# simulation/utils.py
"""工具函数: 实验文件夹管理、配置保存、数据序列化、Checkpoint 参数指纹校验"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import config


# ==============================================================================
# Checkpoint Parameter Fingerprint
# ==============================================================================

def _build_param_fingerprint() -> dict:
    """
    Build a parameter fingerprint from current config.py values.

    This captures ALL parameters that affect simulation behavior.
    When saving a checkpoint, this fingerprint is embedded;
    when resuming, it is compared against the current config to detect mismatches.
    """
    return {
        # Receptor kinetics
        "TAU_ON_D1":    config.TAU_ON_D1,
        "TAU_OFF_D1":   config.TAU_OFF_D1,
        "TAU_ON_D2":    config.TAU_ON_D2,
        "TAU_OFF_D2":   config.TAU_OFF_D2,
        "EC50_D1":      config.EC50_D1,
        "EC50_D2":      config.EC50_D2,
        "BETA":         config.BETA,
        # Modulation strength
        "EPS_D1":       config.EPS_D1,
        "EPS_D2":       config.EPS_D2,
        "BIAS_D1":      config.BIAS_D1,
        "BIAS_D2":      config.BIAS_D2,
        "LAM_D1":       config.LAM_D1,
        "LAM_D2":       config.LAM_D2,
        # LIF parameters
        "V_REST":       config.V_REST,
        "V_RESET":      config.V_RESET,
        "V_TH":         config.V_TH,
        "R_BASE":       config.R_BASE,
        "C_E":          config.C_E,
        "C_I":          config.C_I,
        "TAU_SYN":      config.TAU_SYN,
        "T_REF":        config.T_REF,
        "BG_MEAN":      config.BG_MEAN,
        "BG_STD":       config.BG_STD,
        # DA baseline
        "DA_BASELINE":  config.DA_BASELINE,
        # Network structure
        "N_E":          config.N_E,
        "N_I":          config.N_I,
        "CONN_PROB":    config.CONN_PROB,
        "W_EXC":        config.W_EXC,
        "W_INH":        config.W_INH,
        "RANDOM_SEED":  config.RANDOM_SEED,
    }


def verify_checkpoint_fingerprint(ckpt_data: dict, checkpoint_path: str) -> None:
    """
    Verify that the checkpoint's parameter fingerprint matches current config.

    If the checkpoint was generated with different parameters, raise ValueError
    with a detailed mismatch report. This prevents silent use of stale checkpoints.

    Args:
        ckpt_data       : Loaded checkpoint dict (must contain 'param_fingerprint' key).
        checkpoint_path : Path string for error message display.

    Raises:
        ValueError : If parameters mismatch or fingerprint is missing.
    """
    if 'param_fingerprint' not in ckpt_data:
        raise ValueError(
            f"\n\n❌ CHECKPOINT FINGERPRINT MISSING!\n"
            f"   File: {checkpoint_path}\n"
            f"   This checkpoint was saved without a parameter fingerprint.\n"
            f"   It was likely generated with an older version of the code.\n"
            f"   Please regenerate the checkpoint:\n"
            f"     python main.py --da <DA_LEVEL> --duration <SECONDS> --save-ckpt\n"
        )

    saved_fp = ckpt_data['param_fingerprint']
    current_fp = _build_param_fingerprint()

    mismatches = []
    for key, current_val in current_fp.items():
        if key not in saved_fp:
            mismatches.append(f"  [NEW PARAM]  {key}: current={current_val}, checkpoint=N/A")
            continue
        saved_val = saved_fp[key]
        # Compare with tolerance for floats
        if isinstance(current_val, float) and isinstance(saved_val, float):
            if abs(current_val - saved_val) > 1e-6:
                mismatches.append(f"  [MISMATCH]   {key}: current={current_val}, checkpoint={saved_val}")
        elif current_val != saved_val:
            mismatches.append(f"  [MISMATCH]   {key}: current={current_val}, checkpoint={saved_val}")

    # Also check for params in checkpoint that are no longer in config
    for key in saved_fp:
        if key not in current_fp:
            mismatches.append(f"  [REMOVED]    {key}: checkpoint={saved_fp[key]}, current=N/A")

    if mismatches:
        mismatch_str = "\n".join(mismatches)
        raise ValueError(
            f"\n\n❌ CHECKPOINT PARAMETER MISMATCH!\n"
            f"   File: {checkpoint_path}\n"
            f"   The checkpoint was generated with different parameters than current config.\n"
            f"   Using this checkpoint would produce incorrect/inconsistent results.\n\n"
            f"   Mismatched parameters:\n{mismatch_str}\n\n"
            f"   To fix this, regenerate the checkpoint with current parameters:\n"
            f"     python main.py --da <DA_LEVEL> --duration <SECONDS> --save-ckpt\n"
        )

    print(f"✅ Checkpoint fingerprint verified — all parameters match current config.")


def setup_experiment_folder(tag: str = "", base_dir: str = "outputs") -> Path:
    """
    Create a timestamped experiment folder with a descriptive tag.

    Folder naming: exp_{YYYY-MM-DD_HH-MM-SS}_{tag}
    e.g. exp_2026-04-06_21-30-00_DA2nM_500s
         exp_2026-04-06_21-30-00_resume_DA15nM_100s
         exp_2026-04-06_21-30-00_2stage_DA2-15nM_130s
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if tag:
        folder_name = f"exp_{timestamp}_{tag}"
    else:
        folder_name = f"exp_{timestamp}"
    exp_dir = Path(base_dir) / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Experiment directory: {exp_dir}")
    return exp_dir


def save_args(args_dict: dict, save_dir: Path):
    """将实验参数配置保存为 JSON。"""
    json_ready = {k: str(v) for k, v in args_dict.items()}
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=4, ensure_ascii=False)


def save_raw_data(data: dict, save_dir: Path, filename: str = "raw_data.pkl"):
    """保存原始仿真数据 (Pickle)。"""
    file_path = save_dir / filename
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Raw data saved to {file_path}")


def save_checkpoint(data: dict, da_level: float, duration_s: float,
                    base_dir: str = "checkpoints"):
    """
    Save checkpoint (final_state + config + param_fingerprint) to checkpoints/.

    The parameter fingerprint captures all config values that affect simulation
    behavior. When resuming from this checkpoint, the fingerprint is compared
    against the current config to detect parameter drift.

    File naming: checkpoints/ckpt_DA{da}nM_{duration}s.pkl
    """
    if 'final_state' not in data:
        print("⚠️  No final_state in data, skipping checkpoint save.")
        return None

    ckpt_dir = Path(base_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build descriptive filename (includes BG_MEAN for traceability)
    da_str = f"{da_level}".rstrip('0').rstrip('.')
    dur_str = f"{int(duration_s)}" if duration_s == int(duration_s) else f"{duration_s:.1f}"
    bg_str = f"{config.BG_MEAN:g}"
    filename = f"ckpt_DA{da_str}nM_bg{bg_str}_{dur_str}s.pkl"
    file_path = ckpt_dir / filename

    # Build parameter fingerprint from current config
    fingerprint = _build_param_fingerprint()

    # Save essential data + fingerprint for resuming
    ckpt_data = {
        'config': data['config'],
        'final_state': data['final_state'],
        'param_fingerprint': fingerprint,
    }
    with open(file_path, "wb") as f:
        pickle.dump(ckpt_data, f)

    print(f"💾 Checkpoint saved to {file_path}")
    print(f"   📋 Parameter fingerprint embedded ({len(fingerprint)} params)")
    return file_path
