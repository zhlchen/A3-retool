#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReTool-SFT 数据获取+转换(官方 examples/retool/sft_data_processing.py 的路径适配版)。

首次运行会经 hf-mirror 下载 JoeYing/ReTool-SFT(w13 直连 HF 不通,mirror 可达),
缓存到 slime-run/data/hf_cache,转换后写出 ReTool-SFT.parquet。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/c00954327/slime-run/data/hf_cache")

from datasets import load_dataset  # noqa: E402

ds = load_dataset("JoeYing/ReTool-SFT")["train"]


def convert(sample):
    return {
        "messages": [
            {"role": t["role"], "content": t["content"]} for t in sample["messages"]
        ]
    }


ds = ds.map(convert)
ds.to_parquet("/home/c00954327/slime-run/data/ReTool-SFT.parquet")
print("samples=%d -> ReTool-SFT.parquet" % len(ds))
