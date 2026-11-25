"""wanqing 远程 bash 容器客户端(retool_wanqing 加固版)。

基于 /home/zhouyang/kuaishou/slime/examples/retool_wanqing/remote_sandbox.py
(zhouyang 目录只读,本文件是拷贝后加固,落在 /home/c00954327/retool_wanqing/)。

与原适配版的差异:
  - 内存规格 1Gi → 4Gi(对齐原版 tool_sandbox 的 RLIMIT 4GB;
    注意 request 与 limit 均计入 user_agent 配额);
  - close() 不再向外抛异常(退房失败只告警,靠 TTL 兜底——与 sandbox测试.py 同款);
  - 全局 session 注册表:active_sessions()/close_all_active(),
    作为 asyncio 取消泄漏的兜底(验收项"取消后回收"的后手);
  - base_url / user_agent / image 支持环境变量覆盖(换机器联调用):
    WANQING_BASE_URL / WANQING_USER_AGENT / WANQING_IMAGE。

生命周期与 sandbox测试.py 一致:
create → is_alive 轮询至 ready → execute 探活(swe-rex 就绪延迟最长 ~90s)
→ write_file / execute / read_file → close。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("WANQING_BASE_URL", "http://wanqing-sandbox.internal/gpu-use")
DEFAULT_USER_AGENT = os.environ.get("WANQING_USER_AGENT", "data-agent-rl")
DEFAULT_IMAGE = os.environ.get(
    "WANQING_IMAGE",
    "registry.corp.kuaishou.com/data-agent-train/data-agent@sha256:"
    "a171754ab2cb82b38cc4cf955595e72abdeeb0f4fa07e503f618c4285cb26338",
)
DEFAULT_RESOURCE_LIMIT = {
    "cpu_request": "1",
    "cpu_limit": "1",
    "mem_request": "4Gi",   # 原 tool_sandbox RLIMIT 4GB;1Gi 会 OOM 杀大内存代码
    "mem_limit": "4Gi",
    "disk_request": "2Gi",
    "disk_limit": "2Gi",
}
DEFAULT_TTL_S = 1800
DEFAULT_EXEC_READY_TIMEOUT_S = 180.0
SUCCESS_STATUSES = ("Success", "Sucess")  # 服务端拼写如此,两种都认

# 全局 session 注册表:create 时登记,close 时移除;
# asyncio 取消打不断线程池里的 close 时,这里是回收兜底。
_ACTIVE_SESSIONS: dict[str, "WanqingSandbox"] = {}


def active_sessions() -> list[str]:
    """当前登记在册、未关闭的 session_id 列表(监控/验收用)。"""
    return list(_ACTIVE_SESSIONS.keys())


def close_all_active() -> list[str]:
    """强制关闭所有在册 session,返回成功关闭的 session_id(泄漏兜底)。"""
    closed = []
    for sid in list(_ACTIVE_SESSIONS.keys()):
        sb = _ACTIVE_SESSIONS[sid]
        try:
            sb.close()
            closed.append(sid)
        except Exception:
            pass
    return closed


def post(base_url: str, path: str, payload: dict, timeout: float = 30.0) -> dict:
    """发起 JSON POST,返回响应体。HTTP 错误带上响应正文。"""
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc


class WanqingSandbox:
    """一台 wanqing bash 容器。一个样本的 episode 对应一个实例,多轮复用。"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        image: str = DEFAULT_IMAGE,
        resource_limit: dict | None = None,
        ttl_s: int = DEFAULT_TTL_S,
        exec_ready_timeout_s: float = DEFAULT_EXEC_READY_TIMEOUT_S,
        pvc_mounts: list[dict] | None = None,
        keep: bool = False,
    ) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.image = image
        self.resource_limit = dict(resource_limit or DEFAULT_RESOURCE_LIMIT)
        self.ttl_s = ttl_s
        self.exec_ready_timeout_s = exec_ready_timeout_s
        self.pvc_mounts = list(pvc_mounts or [])
        self.keep = keep
        self.session_id: str | None = None

    def __enter__(self) -> WanqingSandbox:
        self.create()
        self.wait_ready()
        self.wait_exec_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep:
            self.close()

    def create(self) -> None:
        payload = {
            "session_type": "bash",
            "user_agent": self.user_agent,
            "gpu_num": 0,
            "resource_limit": self.resource_limit,
            "image_url": self.image,
            "ttl": self.ttl_s,
        }
        if self.pvc_mounts:
            payload["pvc_name"] = self.pvc_mounts
        data = post(self.base_url, "/create_sandbox", payload, timeout=60)
        if data.get("status") not in SUCCESS_STATUSES:
            raise RuntimeError(f"创建失败: {data.get('error_msg') or data}")
        self.session_id = data.get("session_id")
        if not self.session_id:
            raise RuntimeError(f"创建成功但没有 session_id: {data}")
        _ACTIVE_SESSIONS[self.session_id] = self

    def wait_ready(self, timeout_s: float = 600.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            data = post(self.base_url, "/is_alive", {"session_id": self.session_id}, timeout=15)
            status = data.get("sandbox_status")
            if status == "ready":
                return
            if status == "failed":
                reason = data.get("failure_reason") or data.get("message") or json.dumps(data)
                raise RuntimeError(f"实例进入 failed 状态: {reason}")
            time.sleep(5)
        raise TimeoutError(f"实例在 {timeout_s:.0f}s 内未达到 ready 状态")

    def wait_exec_ready(self) -> None:
        deadline = time.time() + self.exec_ready_timeout_s
        last_error = ""
        while time.time() < deadline:
            try:
                data = self.execute("true", timeout=10.0, http_timeout=30)
                if data["status"] in SUCCESS_STATUSES:
                    return
                last_error = data["error_msg"] or str(data)[:200]
            except Exception as exc:
                last_error = str(exc)[:200]
            time.sleep(5)
        raise TimeoutError(f"执行服务在 {self.exec_ready_timeout_s:.0f}s 内未就绪(最后错误: {last_error})")

    def write_file(self, path: str, content: str) -> None:
        data = post(
            self.base_url,
            "/request/write_file",
            {"session_id": self.session_id, "attach_arg": {"path": path, "content": content}},
            timeout=20,
        )
        if data.get("status") not in SUCCESS_STATUSES:
            raise RuntimeError(f"write_file 失败: {data.get('error_msg') or data}")

    def read_file(self, path: str) -> str:
        data = post(
            self.base_url,
            "/request/read_file",
            {"session_id": self.session_id, "attach_arg": {"path": path}},
            timeout=20,
        )
        if data.get("status") not in SUCCESS_STATUSES:
            raise RuntimeError(f"read_file 失败: {data.get('error_msg') or data}")
        return (data.get("response") or {}).get("content") or ""

    def execute(self, command: str, timeout: float = 120.0, http_timeout: float | None = None) -> dict:
        """同步执行一条 shell 命令,返回 status、stdout、stderr、exit_code。"""
        data = post(
            self.base_url,
            "/request/execute",
            {
                "session_id": self.session_id,
                "attach_arg": {
                    "command": command,
                    "timeout": timeout,
                    "shell": True,
                    "check": False,  # 必须是布尔,字符串会被 400
                },
            },
            timeout=http_timeout if http_timeout is not None else timeout + 15,
        )
        response = data.get("response") or {}
        return {
            "status": data.get("status"),
            "error_msg": data.get("error_msg") or "",
            "stdout": response.get("stdout") or "",
            "stderr": response.get("stderr") or "",
            "exit_code": response.get("exit_code"),
        }

    def close(self) -> None:
        """关闭实例并释放配额。失败只告警不抛(原适配版会抛,TTL 1800s 兜底)。"""
        if not self.session_id:
            return
        sid = self.session_id
        try:
            post(self.base_url, "/close_sandbox", {"session_id": sid}, timeout=15)
        except Exception as exc:
            print(f"[wanqing] close {sid} 失败(将靠 TTL 兜底): {exc}", flush=True)
        finally:
            self.session_id = None
            _ACTIVE_SESSIONS.pop(sid, None)
