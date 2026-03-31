# utils.py
"""工具函数: 实验文件夹管理、配置保存、数据序列化"""
import json
import pickle
from datetime import datetime
from pathlib import Path


def setup_experiment_folder(base_dir: str = "outputs") -> Path:
    """创建带时间戳的实验文件夹, 返回 Path 对象。"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
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
    Save checkpoint (final_state + config) to a shared checkpoints/ folder.

    The file is named with DA level and duration for easy identification, e.g.:
        checkpoints/ckpt_DA2.0nM_500s.pkl

    This allows --resume to reference a stable path that won't change
    between experiment runs.
    """
    if 'final_state' not in data:
        print("⚠️  No final_state in data, skipping checkpoint save.")
        return None

    ckpt_dir = Path(base_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build descriptive filename
    da_str = f"{da_level}".rstrip('0').rstrip('.')
    dur_str = f"{int(duration_s)}" if duration_s == int(duration_s) else f"{duration_s:.1f}"
    filename = f"ckpt_DA{da_str}nM_{dur_str}s.pkl"
    file_path = ckpt_dir / filename

    # Save only the essential data for resuming (much smaller than full raw_data.pkl)
    ckpt_data = {
        'config': data['config'],
        'final_state': data['final_state'],
    }
    with open(file_path, "wb") as f:
        pickle.dump(ckpt_data, f)
    print(f"💾 Checkpoint saved to {file_path}")
    return file_path
