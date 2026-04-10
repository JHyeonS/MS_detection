#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

REPO = Path("/home/ted1204/MS_Detection")

TEXT_REPLACEMENTS = [
    # Python import paths
    ("from src.detection.utils.", "from src.detection.utils."),
    ("from src.detection.dataloader.", "from src.detection.dataloader."),
    ("from src.detection.dataset.", "from src.detection.dataset."),
    ("from src.detection.analysis.", "from src.detection.analysis."),
    ("from src.detection.training.", "from src.detection.training."),
    ("import src.detection.utils.", "import src.detection.utils."),
    ("import src.detection.dataloader.", "import src.detection.dataloader."),
    ("import src.detection.dataset.", "import src.detection.dataset."),
    ("import src.detection.analysis.", "import src.detection.analysis."),
    ("import src.detection.training.", "import src.detection.training."),

    # Shell / launcher paths
    ("python src/detection/training/trainer_pretrain.py", "python src/detection/training/trainer_pretrain.py"),
    ("python src/detection/training/trainer_finetune.py", "python src/detection/training/trainer_finetune.py"),
    ("python src/detection/training/trainer_test.py", "python src/detection/training/trainer_test.py"),
    ('pkill -f "src/detection/training/trainer_pretrain.py"', 'pkill -f "src/detection/training/trainer_pretrain.py"'),
    ('pkill -f "src/detection/training/trainer_finetune.py"', 'pkill -f "src/detection/training/trainer_finetune.py"'),
    ('pkill -f "src/detection/training/trainer_test.py"', 'pkill -f "src/detection/training/trainer_test.py"'),

    # Config paths
    ("configs/train/", "configs/train/"),
]

TARGET_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".txt"}

def should_edit(path: Path) -> bool:
    return path.is_file() and path.suffix in TARGET_SUFFIXES

def main():
    changed = []
    for path in REPO.rglob("*"):
        if not should_edit(path):
            continue
        if any(part in {".git", "__pycache__", "runs", "logs", "outputs", "output", "output_npy"} for part in path.parts):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        new_text = text
        for old, new in TEXT_REPLACEMENTS:
            new_text = new_text.replace(old, new)

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed.append(str(path))

    print(f"[DONE] changed files: {len(changed)}")
    for p in changed:
        print(p)

if __name__ == "__main__":
    main()