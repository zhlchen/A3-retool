"""无网络自检:to_thread 包装 + 解析 + 失败语义 + 输出截断(Phase 2 Step 1 冒烟)。

不需要 wanqing 网络、不加载模型:用假解释器验证 generate_with_wanqing 的
异步接线和 retool_style 加固语义。跑法(容器内,需完整 PYTHONPATH):

    docker exec czl_slime bash -lc 'cd /home/c00954327/slime-ascend/slime && \
    PYTHONPATH=/home/c00954327/Megatron-LM:/home/c00954327/Megatron-Bridge/src:\
/home/c00954327/mbridge:/home/c00954327/sglang/python:\
/home/c00954327/slime-ascend/slime:\
/home/c00954327/slime-ascend/slime/examples/retool:\
/home/c00954327/retool_wanqing \
    python3 /home/c00954327/retool_wanqing/test_async_wrapper.py'
"""
import asyncio

from generate_with_wanqing import execute_predictions
from retool_style import MAX_OUTPUT_CHARS, _truncate


class _SB:
    session_id = "fake-session-1"


class FakeInterpreter:
    sandbox = _SB()
    last_exit_code = 0
    last_latency_ms = 12.3
    last_error = ""

    def execute_code(self, code):
        return "Output:\n42"


async def main():
    # 1. to_thread 包装 + <code> 解析 + <interpreter> 包裹
    obs, done = await execute_predictions("<code>\nprint(6 * 7)\n</code>", FakeInterpreter(), sid="selftest", turn=1)
    assert "42" in obs and "<interpreter>" in obs and done is False, obs

    # 2. 答案即终止
    obs, done = await execute_predictions("Answer: \\boxed{42}", FakeInterpreter(), sid="selftest", turn=2)
    assert obs == "" and done is True

    # 3. 失败语义:容器不可用 -> 错误文本回灌,不抛异常
    obs, done = await execute_predictions("<code>\nprint(1)\n</code>", None, sid="selftest", turn=3)
    assert "Error: sandbox unavailable" in obs and done is False, obs

    # 4. 输出截断
    t = _truncate("x" * (MAX_OUTPUT_CHARS + 100))
    assert t.startswith("x" * 1000) and "truncated" in t

    print("async wrapper + failure semantics + truncation: OK")


asyncio.run(main())
