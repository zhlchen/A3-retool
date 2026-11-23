#!/usr/bin/env python3
"""One-shot DAPO converter: verl-format or slime-format -> retool training format.

Usage:
    python3 fix_dapo_all.py <input.jsonl> <output.jsonl>

Accepts either source shape (auto-detected per record):
  - verl raw (BytedTsinghua-SIA/DAPO-Math-17k):
      prompt = [{'content': ...}, ...] (chat messages), answer inside reward_model.ground_truth
  - already-labeled (zhuzilin/dapo-math-17k style or our fix_dapo_label.py output):
      prompt may be a string or a message list, top-level label may exist

Output record: {"prompt": <plain string>, "label": <ground-truth string>}
"""
import json
import sys


def to_plain_prompt(prompt):
    if isinstance(prompt, list):
        # take the first user message's content (official rl_data_preprocess.py behavior)
        for m in prompt:
            if m.get("role") == "user":
                return m["content"]
        return prompt[0]["content"] if prompt else None
    return prompt


def to_label(d):
    if isinstance(d.get("label"), (str, int, float)) and str(d["label"]).strip():
        return str(d["label"])
    rm = d.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if gt is not None:
        return str(gt)
    raise KeyError(f"record has neither top-level label nor reward_model.ground_truth: keys={list(d.keys())[:8]}")


def main(src, dst):
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out = {"prompt": to_plain_prompt(d.get("prompt")), "label": to_label(d)}
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"converted {n} records -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
