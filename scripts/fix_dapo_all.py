#!/usr/bin/env python3
"""One-shot DAPO converter: accepts a data FILE or a DATASET DIRECTORY.

Usage:
    python3 fix_dapo_all.py <input.jsonl | input.parquet | dataset_dir> <output.jsonl>

If given a directory, auto-finds the data file inside:
  - *.parquet (preferred, read via pandas)
  - *.jsonl / *.json (jsonl-lines; dataset_infos.json and other metadata are skipped)

Accepts either source shape (auto-detected per record):
  - verl raw (BytedTsinghua-SIA/DAPO-Math-17k):
      prompt = [{'content': ...}, ...] (chat messages), answer inside reward_model.ground_truth
  - already-labeled (zhuzilin/dapo-math-17k style or fix_dapo_label.py output):
      prompt may be a string or a message list, top-level label may exist

Output record: {"prompt": <plain string>, "label": <ground-truth string>}
"""
import glob
import json
import os
import sys


def to_plain_prompt(prompt):
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content")
        return prompt[0].get("content") if prompt else None
    return prompt


def to_label(d):
    v = d.get("label")
    if isinstance(v, (str, int, float)) and str(v).strip():
        return str(v)
    rm = d.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if gt is not None:
        return str(gt)
    raise KeyError(f"record has neither top-level label nor reward_model.ground_truth: keys={list(d.keys())[:8]}")


def find_data_file(target):
    if os.path.isfile(target):
        return target
    if os.path.isdir(target):
        for pat in ("*.parquet", "*.jsonl", "*.json"):
            hits = [p for p in sorted(glob.glob(os.path.join(target, "**", pat), recursive=True))
                    if "dataset_infos" not in os.path.basename(p)]
            if hits:
                print(f"[info] directory given, using data file: {hits[0]}")
                return hits[0]
        raise FileNotFoundError(f"no .parquet/.jsonl/.json data file found under {target}")
    raise FileNotFoundError(target)


def iter_records(path):
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield {k: (v.tolist() if hasattr(v, "tolist") and not isinstance(v, (str, bytes)) else v)
                   for k, v in row.items()}
    else:
        with open(path) as fi:
            for line in fi:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(target, dst):
    path = find_data_file(target)
    n = 0
    with open(dst, "w") as fo:
        for d in iter_records(path):
            out = {"prompt": to_plain_prompt(d.get("prompt")), "label": to_label(d)}
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"converted {n} records -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
