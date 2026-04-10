#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def show_one(csv_path, label_col="label_name", source_col="site"):
    df = pd.read_csv(csv_path)
    print(f"\n===== {csv_path} =====")
    print(f"rows: {len(df)}")

    if label_col in df.columns:
        print("\n[label counts]")
        print(df[label_col].value_counts(dropna=False))

        print("\n[label ratio]")
        print(df[label_col].value_counts(normalize=True, dropna=False))

    if source_col in df.columns:
        print(f"\n[{source_col} counts]")
        print(df[source_col].value_counts(dropna=False))

    if label_col in df.columns and source_col in df.columns:
        print(f"\n[{source_col} x {label_col}]")
        print(pd.crosstab(df[source_col], df[label_col]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="label_name")
    ap.add_argument("--source_col", type=str, default="site")
    args = ap.parse_args()

    show_one(args.train_csv, args.label_col, args.source_col)
    show_one(args.val_csv, args.label_col, args.source_col)
    show_one(args.test_csv, args.label_col, args.source_col)

if __name__ == "__main__":
    main()