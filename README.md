# A3-retool

slime retool(工具调用 RL)在 Ascend 910B3 上的落地,以及 **wanqing 远程沙箱执行端适配**。

本仓库核心是 `retool_wanqing/`:把 slime `examples/retool` 的 `code_interpreter`
执行层从**本机 subprocess 沙箱**(`tool_sandbox.py`)换成 **wanqing 远程 bash
容器**(HTTP 生命周期管理)。多轮循环、四级解析、loss mask 规则、奖励函数与
原版逐行一致——**只换执行这一格**,训练语义零漂移。

适配基线:slime 上游 18868a79 代的 `examples/retool`(与 910B3 训练栈
slime-ascend 同代)。执行端适配本身与硬件无关。

## 仓库结构

```
retool_wanqing/                    wanqing 远程沙箱执行端(本仓库核心)
    ├── remote_sandbox.py           HTTP 客户端:一台容器全生命周期封装
    ├── retool_style.py             retool 四级解析 + RemoteCodeInterpreter
    ├── generate_with_wanqing.py    训练接线核心:多轮循环 × 远程执行
    ├── run_test.py                 无模型的接口测试驱动
    └── test_async_wrapper.py       无网络自检(异步/失败语义/截断)
scripts/                            910B3 训练启动与数据脚本(裸 GRPO / retool RL / SFT)
README-冒烟结果.md                   冒烟与长跑结果记录(11-18 裸 GRPO、11-20 retool、11-23 31 轮长跑)
```

## 架构与分岔点

依赖单向向下:

```
generate_with_wanqing.generate      训练侧入口(train.py --custom-generate-function-path)
    │ import 解析/模板/奖励
    ▼
retool_style(四级解析 + RemoteCodeInterpreter)
    │ import
    ▼
remote_sandbox(WanqingSandbox:HTTP 客户端)
    │ HTTP
    ▼
wanqing 服务(k8s bash 容器)
```

与原版 retool 的分岔点在 `execute_predictions` 的**解释器注入**:
原版写死全局 `tool_registry`(本机 subprocess);本适配把它做成参数,
注入 `RemoteCodeInterpreter`(远程容器)。模板、四级解析、奖励直接
`import` 自 `generate_with_retool`,不复制——单一事实源,解析行为与
已验证的本机沙箱路线保持逐字一致。

## 设计要点(为什么这些代码长这样)

| 机制 | 原因 |
| --- | --- |
| 所有阻塞调用经 `asyncio.to_thread` 下放线程池 | 原版 `execute_predictions` 是 async;同步 urllib 直接调用会**卡死事件循环**(建容器/等就绪/探活/执行/关容器全部下放) |
| 每样本一台容器,多轮复用,`finally` + `asyncio.shield` 关闭 | 与官方语义一致(episode 结束即还配额);正常/异常/取消三条路都退房;另有全局 session 注册表(`active_sessions`/`close_all_active`)作泄漏兜底 |
| 失败一律返回错误文本回灌,不抛异常 | 容器建不起来/执行失败/超时 → "Error: ..." 包成 interpreter 观察(与原版本机沙箱同款),样本按现状收尾,不炸整个 step |
| `asyncio.Semaphore(WANQING_MAX_CONTAINERS, 16)` 容器并发闸 | 同时存活的容器数受控,保护 user_agent 配额(服务侧按实例数限额) |
| 执行超时 120s / 内存 4Gi | 对齐原版 tool_sandbox 预算(早期适配版误设 30s/1Gi,已修) |
| 输出截断(默认 10000 字符) | 防模型代码海量 print 撑爆回灌上下文 |
| 逐调用日志 | `[wanqing] sid/session/turn/exit/latency/out_chars` 一行,create/close 各一行配对,`log_summary()` 出累计统计——样本 ID、session ID、执行结果三方可关联 |

## 使用

三级测试阶梯(由轻到重):

```bash
# 1. 无网络自检:异步包装/解析/失败语义/截断(不需要 wanqing,不需要模型)
python3 test_async_wrapper.py

# 2. 接口级测试:解析 + 远程执行 + 回读校验(需要能解析服务域名的网络,不需要模型)
python3 run_test.py
python3 run_test.py --code 'print(6 * 7)'

# 3. 接入训练:slime 启动脚本改两处
#      PYTHONPATH 追加本目录
#      --custom-generate-function-path generate_with_wanqing.generate
#    (--custom-rm-path 保持 generate_with_retool.reward_func 不变)
```

训练侧完整流程(数据准备、RL/SFT 启动、冒烟结果)见 `scripts/` 与
`README-冒烟结果.md`。

## 配置(环境变量)

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `WANQING_BASE_URL` | `http://wanqing-sandbox.internal/gpu-use` | 服务地址 |
| `WANQING_USER_AGENT` | `data-agent-rl` | 配额主体(需服务方开通) |
| `WANQING_IMAGE` | data-agent 镜像 digest | 容器镜像(须预入库) |
| `WANQING_MAX_CONTAINERS` | `16` | 同时存活的容器数上限 |
| `WANQING_MAX_OUTPUT_CHARS` | `10000` | 工具输出截断长度 |

## 状态

- **离线自检:通过**(训练容器内,含 slime 全链 import)
- 接口级(run_test):待网络(训练机到服务的 DNS + 路由开通中)
- RL 端到端:待网络后以 2 并发轨验证(文件隔离/回收/日志关联)

## 服务侧约定(已内置处理的坑)

两阶段就绪(容器 ready 后执行服务仍有 ~90s 延迟,客户端自动探活);
`execute` 的 `check` 字段必须传布尔(字符串会被 400);服务端把 Success
拼成 "Sucess"(两种都认);close 失败只告警,靠 TTL(默认 1800s)兜底回收。
