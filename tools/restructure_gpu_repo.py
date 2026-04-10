#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe 1st-stage restructure for MS_Detection repo on GPU server.

Usage:
    python tools/restructure_gpu_repo.py --repo_root /home/ted1204/MS_Detection

Behavior:
- Creates new Codex-friendly directory layout
- Moves known source/configs/train/script directories into new structure
- Backs up overwritten targets if needed
- Does NOT touch runs/, logs/, outputs/, data/ etc.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, required=True, help="Path to MS_Detection repo root")
    parser.add_argument("--dry_run", action="store_true", help="Print actions only")
    return parser.parse_args()


def safe_mkdir(path: Path, dry_run: bool):
    if not path.exists():
        print(f"[MKDIR] {path}")
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path, dry_run: bool):
    if not src.exists():
        print(f"[SKIP] source not found: {src}")
        return

    if dst.exists():
        backup_name = dst.parent / f"{dst.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[BACKUP] {dst} -> {backup_name}")
        if not dry_run:
            shutil.move(str(dst), str(backup_name))

    print(f"[MOVE] {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def write_text(path: Path, content: str, dry_run: bool):
    if path.exists():
        print(f"[SKIP] file exists: {path}")
        return
    print(f"[WRITE] {path}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    dry_run = args.dry_run

    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    print("=" * 80)
    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] dry_run  : {dry_run}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Create target directory structure
    # ------------------------------------------------------------------
    target_dirs = [
        repo_root / "src" / "detection" / "dataloader",
        repo_root / "src" / "detection" / "dataset",
        repo_root / "src" / "detection" / "training",
        repo_root / "src" / "detection" / "analysis",
        repo_root / "src" / "detection" / "utils",
        repo_root / "src" / "models",
        repo_root / "src" / "shared",
        repo_root / "scripts" / "gpu",
        repo_root / "configs" / "train",
        repo_root / "configs" / "experiments",
        repo_root / "configs" / "system",
        repo_root / "env",
        repo_root / "docs",
        repo_root / "tools",
    ]
    for d in target_dirs:
        safe_mkdir(d, dry_run)

    # ------------------------------------------------------------------
    # 2. Move known existing source dirs into new structure
    # ------------------------------------------------------------------
    move_map = [
        (repo_root / "src" / "dataloader", repo_root / "src" / "detection" / "dataloader"),
        (repo_root / "src" / "dataset", repo_root / "src" / "detection" / "dataset"),
        (repo_root / "src" / "training", repo_root / "src" / "detection" / "training"),
        (repo_root / "src" / "analysis", repo_root / "src" / "detection" / "analysis"),
        (repo_root / "src" / "utils", repo_root / "src" / "detection" / "utils"),
        (repo_root / "src" / "models", repo_root / "src" / "models"),
    ]

    for src, dst in move_map:
        if src.resolve() == dst.resolve():
            continue
        safe_move(src, dst, dry_run)

    # ------------------------------------------------------------------
    # 3. Move config directory into configs/train if old config exists
    # ------------------------------------------------------------------
    old_config = repo_root / "config"
    new_train_config = repo_root / "configs" / "train"
    if old_config.exists():
        # move contents instead of whole directory name conflict
        for item in old_config.iterdir():
            safe_move(item, new_train_config / item.name, dry_run)

        print(f"[INFO] old config directory handled: {old_config}")

    # ------------------------------------------------------------------
    # 4. Move scripts into scripts/gpu (except already nested)
    # ------------------------------------------------------------------
    old_scripts = repo_root / "scripts"
    new_gpu_scripts = repo_root / "scripts" / "gpu"

    if old_scripts.exists():
        for item in list(old_scripts.iterdir()):
            if item.name == "gpu":
                continue
            if item.name == "cpu":
                continue
            safe_move(item, new_gpu_scripts / item.name, dry_run)

    # ------------------------------------------------------------------
    # 5. Write starter files if missing
    # ------------------------------------------------------------------
    gitignore_text = """__pycache__/
*.pyc
*.pyo
*.pyd

.venv/
venv/
env/

.ipynb_checkpoints/

.DS_Store
Thumbs.db

logs/
*.log
*.err
*.out

runs/
outputs/
checkpoints/
wandb/

*.pt
*.pth
*.ckpt

data/
datasets/
raw_data/
output/
output_npy/
old_outputs/

*.npy
*.npz
*.h5
*.hdf5
*.tdms
*.sgy
*.segy
*.gz
*.[0-9]

*.png
*.jpg
*.jpeg
"""

    agents_text = """# AGENTS.md

## Project structure
- src/detection: training / evaluation pipeline
- src/models: model definitions
- scripts/gpu: GPU-side entry scripts
- configs/train: training configs
- configs/experiments: experiment-specific configs
- configs/system: machine-specific path configs

## Rules
- Do not commit runs/, logs/, outputs/, checkpoints/, data/, output_npy/
- Prefer config-driven paths over hardcoded absolute paths
- New GPU launchers go into scripts/gpu
- Keep training code under src/detection/training
- Keep dataloader code under src/detection/dataloader
- Keep dataset code under src/detection/dataset
- Use executable Python scripts when possible
"""

    gpu_system_yaml = """paths:
  repo_root: /home/ted1204/MS_Detection
  run_root: /home/ted1204/MS_Detection/runs
  checkpoint_root: /home/ted1204/MS_Detection/checkpoints
  npy_root: /home/ted1204/MS_Detection/data
  log_root: /home/ted1204/MS_Detection/logs
"""

    env_gpu_yaml = """name: ms_detection
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1
  - numpy
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - pyyaml
  - tqdm
  - pip
"""

    write_text(repo_root / ".gitignore", gitignore_text, dry_run)
    write_text(repo_root / "AGENTS.md", agents_text, dry_run)
    write_text(repo_root / "configs" / "system" / "gpu_server.yaml", gpu_system_yaml, dry_run)
    write_text(repo_root / "env" / "environment_gpu.yml", env_gpu_yaml, dry_run)

    print("=" * 80)
    print("[DONE] GPU repo restructure scaffold complete")
    print("=" * 80)


if __name__ == "__main__":
    main()