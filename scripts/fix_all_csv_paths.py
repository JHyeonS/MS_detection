#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_path_column(df: pd.DataFrame) -> str | None:
    for col in ["npy_path", "path"]:
        if col in df.columns:
            return col
    return None


def fix_one_csv(
    csv_path: Path,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = False,
    make_backup: bool = True,
) -> tuple[int, int]:
    """
    Returns:
        changed_rows, existing_rows_after_replace
    """
    df = pd.read_csv(csv_path)
    path_col = find_path_column(df)

    if path_col is None:
        print(f"[SKIP] no path column: {csv_path}")
        return 0, 0

    old_prefix = str(Path(old_prefix))
    new_prefix = str(Path(new_prefix))

    original_paths = df[path_col].astype(str).tolist()
    updated_paths = []

    changed = 0
    exists_count = 0

    for p in original_paths:
        if p.startswith(old_prefix):
            new_p = new_prefix + p[len(old_prefix):]
            changed += 1
        else:
            new_p = p

        updated_paths.append(new_p)

        if Path(new_p).exists():
            exists_count += 1

    print(f"[CSV ] {csv_path}")
    print(f"      path_col={path_col} | rows={len(df)} | changed={changed} | exists_after={exists_count}/{len(df)}")

    if changed > 0:
        preview_n = min(3, len(df))
        for i in range(preview_n):
            if original_paths[i] != updated_paths[i]:
                print(f"      BEFORE: {original_paths[i]}")
                print(f"      AFTER : {updated_paths[i]}")

    if dry_run:
        return changed, exists_count

    if changed > 0:
        if make_backup:
            backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(csv_path, backup_path)

        df[path_col] = updated_paths
        df.to_csv(csv_path, index=False)

    return changed, exists_count


def main():
    parser = argparse.ArgumentParser(
        description="Recursively fix path prefixes in all CSV files under a directory."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory to search CSV files recursively",
    )
    parser.add_argument(
        "--old_prefix",
        type=str,
        required=True,
        help="Old absolute path prefix",
    )
    parser.add_argument(
        "--new_prefix",
        type=str,
        required=True,
        help="New absolute path prefix",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview only, do not modify files",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create .bak backup files",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    csv_files = sorted(root_dir.rglob("*.csv"))
    if len(csv_files) == 0:
        print(f"[INFO] No CSV files found under: {root_dir}")
        return

    print(f"[INFO] root_dir    : {root_dir}")
    print(f"[INFO] old_prefix : {args.old_prefix}")
    print(f"[INFO] new_prefix : {args.new_prefix}")
    print(f"[INFO] csv_count   : {len(csv_files)}")
    print(f"[INFO] dry_run    : {args.dry_run}")
    print()

    total_changed = 0
    total_csv_changed = 0

    for csv_path in csv_files:
        changed, _ = fix_one_csv(
            csv_path=csv_path,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            dry_run=args.dry_run,
            make_backup=not args.no_backup,
        )
        total_changed += changed
        if changed > 0:
            total_csv_changed += 1

    print()
    print(f"[DONE] scanned csv files : {len(csv_files)}")
    print(f"[DONE] changed csv files : {total_csv_changed}")
    print(f"[DONE] changed rows total: {total_changed}")

    if args.dry_run:
        print("[DRY-RUN] No files were modified.")


if __name__ == "__main__":
    main()