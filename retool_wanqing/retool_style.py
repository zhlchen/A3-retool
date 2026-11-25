"""用 retool 的工具调用格式驱动一台 wanqing 容器(加固版)。

模型文本的解析顺序与 examples/retool/generate_with_retool.py 相同(勿改,保持 bug 级兼容)。

与原适配版的差异:
  - 执行超时 30s → 120s(对齐原版 tool_sandbox 的 python_timeout);
  - execute_code 不再抛异常:写/执行失败一律返回错误文本回灌
    (与原版 tool_sandbox「错误回灌不炸训练」同款语义);
  - 工具输出截断:WANQING_MAX_OUTPUT_CHARS(默认 10000)字符;
  - 记录 last_exit_code / last_latency_ms / last_error 供上层日志关联。
"""

from __future__ import annotations

import json
import os
import re
import time

from remote_sandbox import WanqingSandbox

TOOL_NAME = "code_interpreter"
MAX_OUTPUT_CHARS = int(os.environ.get("WANQING_MAX_OUTPUT_CHARS", "10000"))
EXEC_TIMEOUT_S = 120.0  # 对齐原版 tool_sandbox python_timeout=120(原适配版误设 30)


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """从一段模型输出里抽出 answer 或 code。命中即返回。"""
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            json_str = tool_call_match.group(1).replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            if tool_call_data.get("name") == TOOL_NAME:
                code = tool_call_data.get("arguments", {}).get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            pass

    code_match = re.search(r"<code>(.*?)</code>", prediction, re.DOTALL)
    if code_match and code_match.group(1).strip():
        return "code", code_match.group(1).strip()

    python_match = re.search(r"```python\s*(.*?)\s*```", prediction, re.DOTALL)
    if python_match and python_match.group(1).strip():
        return "code", python_match.group(1).strip()

    return None, ""


class RemoteCodeInterpreter:
    """retool 的 code_interpreter。一次调用对应容器里的一次 python3。"""

    def __init__(self, sandbox: WanqingSandbox) -> None:
        self.sandbox = sandbox
        self.turn = 0
        self.last_exit_code = None
        self.last_latency_ms = 0.0
        self.last_error = ""

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name != TOOL_NAME:
            return f"Error: Tool '{tool_name}' not implemented"
        code = arguments.get("code", "")
        if not str(code).strip():
            return "Error: No code provided"
        return self.execute_code(str(code))

    def execute_code(self, code: str) -> str:
        """加固语义:任何失败不抛异常,返回错误文本回灌;结果记 last_* 供日志层。"""
        self.turn += 1
        self.last_exit_code = None
        self.last_error = ""
        path = f"/tmp/retool_code_{self.turn}.py"
        t0 = time.time()
        try:
            self.sandbox.write_file(path, code if code.endswith("\n") else code + "\n")
            result = self.sandbox.execute(f"python3 {path}", timeout=EXEC_TIMEOUT_S)
        except Exception as exc:  # 网络/服务层失败:错误回灌,不炸训练
            self.last_latency_ms = (time.time() - t0) * 1000.0
            self.last_error = str(exc)[:300]
            return f"Error: sandbox execute failed: {self.last_error}"
        self.last_latency_ms = (time.time() - t0) * 1000.0
        self.last_exit_code = result["exit_code"]
        if result["status"] not in ("Success", "Sucess"):
            self.last_error = (result["error_msg"] or result["status"])[:300]
            return f"Error: execute failed: {self.last_error}"
        return _format_process_result(result["stdout"], result["stderr"], result["exit_code"])


def execute_predictions(prediction: str, interpreter: RemoteCodeInterpreter) -> tuple[str, bool]:
    """复现 retool 的 execute_predictions：代码进容器，答案结束本轮。"""
    action, content = postprocess_predictions(prediction)
    if action == "code":
        result = interpreter.execute_tool(TOOL_NAME, {"code": content.strip()})
        return f"\n\n<interpreter>\n{result}\n</interpreter>\n\n", False
    if action == "answer":
        return "", True
    return (
        "\nMy previous action is invalid. "
        "If I want to execute code, I should put the code between <code> and </code>. "
        "If I want to give the final answer, I should use the format "
        "'Answer: \\boxed{answer}'. Let me try again.\n"
    ), False


def _format_process_result(stdout: str, stderr: str, exit_code: int | None) -> str:
    stdout = _truncate(stdout.strip())
    stderr = _truncate(stderr.strip())
    if exit_code not in (0, None):
        details = stderr or stdout or "(no output)"
        return f"Error: Process exited with code {exit_code}\n{details}"
    parts = []
    if stdout:
        parts.append(f"Output:\n{stdout}")
    if stderr:
        parts.append(f"Errors:\n{stderr}")
    return "\n".join(parts)


def _truncate(text: str) -> str:
    """截断工具输出,防模型代码海量 print 撑爆回灌上下文。"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
