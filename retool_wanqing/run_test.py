#!/usr/bin/env python3
"""按 retool 的工具调用方式，在一台 wanqing 容器里跑几段 Python。

不启动模型和训练。每段「模型输出」都走和 generate_with_retool.execute_predictions
相同的解析，代码通过 write_file + execute 送到远程 bash 容器。

    python3 run_test.py
    python3 run_test.py --code 'print(6 * 7)'
    python3 run_test.py --keep
"""

from __future__ import annotations

import argparse
import sys

from remote_sandbox import (
    DEFAULT_BASE_URL,
    DEFAULT_EXEC_READY_TIMEOUT_S,
    DEFAULT_IMAGE,
    DEFAULT_RESOURCE_LIMIT,
    DEFAULT_TTL_S,
    DEFAULT_USER_AGENT,
    WanqingSandbox,
)
from retool_style import RemoteCodeInterpreter, execute_predictions, postprocess_predictions

DEFAULT_TURNS = [
    '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(6 * 7)"}}\n</tool_call>',
    "<code>\nprint('hello from retool_wanqing')\n</code>",
    r"Answer: \boxed{42}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 retool 工具调用格式测试 wanqing 容器")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--ttl-s", type=int, default=DEFAULT_TTL_S)
    parser.add_argument("--exec-ready-timeout-s", type=float, default=DEFAULT_EXEC_READY_TIMEOUT_S)
    parser.add_argument(
        "--pvc",
        action="append",
        default=None,
        metavar="NAME:SUB_PATH:MOUNT_PATH",
        help="只读挂载 PVC，可重复。MOUNT_PATH 必须是目录",
    )
    parser.add_argument("--code", default=None, help="跳过预制轮次，只把这一段 Python 当 code_interpreter 执行")
    parser.add_argument("--keep", action="store_true", help="结束后保留容器，等待 TTL 回收")
    return parser.parse_args()


def parse_pvc(specs: list[str] | None) -> list[dict]:
    mounts = []
    for spec in specs or []:
        name, sub_path, mount_path = spec.split(":", 2)
        mounts.append({"name": name, "path": mount_path, "sub_path": sub_path, "read_only": True})
    return mounts


def main() -> int:
    args = parse_args()
    turns = [f"<code>\n{args.code}\n</code>"] if args.code is not None else list(DEFAULT_TURNS)
    sandbox = WanqingSandbox(
        base_url=args.base_url,
        user_agent=args.user_agent,
        image=args.image,
        resource_limit=DEFAULT_RESOURCE_LIMIT,
        ttl_s=args.ttl_s,
        exec_ready_timeout_s=args.exec_ready_timeout_s,
        pvc_mounts=parse_pvc(args.pvc),
        keep=args.keep,
    )
    print(f"服务地址: {args.base_url}")
    print(f"配额主体: {args.user_agent}")
    try:
        with sandbox:
            print(f"容器已就绪 session_id={sandbox.session_id}")
            interpreter = RemoteCodeInterpreter(sandbox)
            observations = []
            for index, prediction in enumerate(turns, start=1):
                action, _content = postprocess_predictions(prediction)
                observation, done = execute_predictions(prediction, interpreter)
                observations.append((observation, done))
                print(f"\n--- turn {index} action={action} done={done} ---")
                print(prediction)
                if observation:
                    print(observation)
                if done:
                    break
            if args.code is None:
                first, second, third = observations
                if "42" not in first[0] or first[1]:
                    raise RuntimeError(f"第一轮代码结果异常: {first[0]!r}")
                if "hello from retool_wanqing" not in second[0] or second[1]:
                    raise RuntimeError(f"第二轮代码结果异常: {second[0]!r}")
                if not third[1]:
                    raise RuntimeError("最终答案没有结束循环")
                written = sandbox.read_file("/tmp/retool_code_1.py")
                if "print(6 * 7)" not in written:
                    raise RuntimeError(f"回读第一段代码失败: {written[:200]}")
                print("read_file /tmp/retool_code_1.py 与写入一致")
            elif not observations or observations[0][1]:
                raise RuntimeError("传入的代码没有被执行")
        print("\n测试通过" if not args.keep else f"\n测试通过，容器保留 session_id={sandbox.session_id}")
        return 0
    except Exception as exc:
        print(f"\n测试失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
