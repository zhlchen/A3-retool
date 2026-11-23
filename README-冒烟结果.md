# Qwen3.5-9B GRPO 冒烟结果与产物说明(2026-11-18,北京)

任务:「完成沙箱对接,打通 910b 上的 slime 流程」——本目录是 11-18 当天打通冒烟的交付物。

> **2026-11-20 追记:retool 沙箱冒烟也已通过**(方向 A:本地 tool_sandbox.py),详见文末「retool 沙箱冒烟」一章;新增交付 logs/retool_smoke.log、scripts/run-qwen3.5-9B-npu-retool.sh、scripts/fix_dapo_retool.py、scripts/test_tool_sandbox.py。

## 结论

**冒烟通过**:910B3(8 卡)上 slime-ascend 的 GRPO 全链路(数据 → SGLang-NPU rollout → deepscaler 奖励 → Megatron-NPU 训练 → 闭环)端到端跑通 2 轮。唯一未落袋:训练结束时的最终 ckpt 存盘因磁盘写满中止(/home 当时剩 13G,单个 ckpt 约 50G)——容量问题,非链路问题。

## 环境

- 机器:w13(178.110.22.6),8×Ascend 910B3(每卡 64G HBM),宿主内存 2T
- 容器:czl_slime(quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12)
- 栈:slime-ascend(gitcode main@e58ab12,patch 管理版,slime 0.4.0)+ sglang v0.5.21+官方 6 补丁 + Megatron-LM 4b4acac9+2 补丁 + mbridge + Megatron-Bridge + MegatronAdaptor + TransformerEngineNPU + sgl-kernel-npu 预编译 wheel + triton-ascend 3.2.1 + flash-linear-attention 0.6.0
- 模型:Qwen3.5-9B(/home/zrw/Qwen3.5-9B,19G BF16,多模态检查点)→ 转换 torch_dist 17G(/home/c00954327/slime-run/Qwen3.5-9B_torch_dist,已缓存,重跑自动跳过)
- 数据:dapo-math-17k(取自 /home/zhouyang/datasets,verl 格式,补 label 后 179 万条)

## 冒烟关键数字(见 logs/smoke_run.log)

- 第 1 轮 rollout:32 样本,生成 73 秒,吞吐 112 tok/GPU/s
- 训练步 1:3 microbatch,2 分 32 秒
- 第 2 轮 rollout + 训练步 2:3 microbatch,15 秒(热身后)
- rewards 0.0:全部样本 2048 token 截断未出答案(基线模型+长推理+temperature 0.8 的正常现象,RL 的起点即如此)
- 权重加载:8 rank(TP4×DP2,`--colocate` 与 sglang TP4 引擎共置 8 卡)

## 交付物清单

| 文件 | 说明 |
|---|---|
| `scripts/run-qwen3.5-9B-npu.sh` | 运行脚本(冒烟/正式双档;与 w13 上 /home/c00954327/slime-ascend/scripts/ 同份,md5 93ab0585) |
| `scripts/quick_install.sh.modified` | 改造版依赖安装器(可断点续装;官方原版可 `git checkout` 还原) |
| `scripts/fix_dapo_label.py` | dapo 数据补 label 转换器(verl 格式→slime 格式) |
| `logs/smoke_run.log` | 冒烟全程日志(含两轮生成样本、训练指标) |
| `logs/train_log_qwen3.5_9b.log` | 脚本 tee 的训练日志 |
| `logs/quick_install.log` | 依赖安装全程日志 |

## 重跑方法(w13)

```
# 冒烟(转换已缓存,约 20 分钟):
docker exec czl_slime bash -lc 'SMOKE_MODE=1 bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu.sh'
# 正式(先解决磁盘,见下):
docker exec czl_slime bash -lc 'bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu.sh'
```

## 一路排掉的坑(按序,均有对应修复)

1. **github 从 w13 断连**(TCP 15s+、传输中断)→ 5 个 github 仓走「本机克隆→tar .git 中转」;zip 走 gh-proxy.com 镜像;gitcode/pypi/OBS 直连正常
2. atb set_env.sh 报 ZSH_VERSION/$1 unbound → 安装器去 nounset
3. git am 缺提交身份 → 容器内 git config(slime-sync)
4. memfabric 1.2.1 URL 404(官方 OBS 删了)→ 换 v1.3.1
5. **TransformerEngineNPU 钉死的 commit 在 gitcode 被镜像同步弄丢** → 换当日 main 头 2c5e511a(TE 无补丁纯安装)
6. unzip 撞 /tmp 残留弹交互提示拖死 set -e → `unzip -o`;`ln -s`→`ln -sf`
7. **sunabako(Rust 沙箱运行时)x86_64-only**(compile_error 门+x86 syscall)→ aarch64 装不了 → 装占位包满足依赖(仅 coding_agent_rl 例子用;本任务沙箱走纯 Python 的 tool_sandbox.py,无影响)
8. 缺 flash-linear-attention(qwen3_5 插件吞了 ImportError)→ pip 装 0.6.0
9. 缺 libGL.so.1(megatron 异步存盘的 mp 子进程 import cv2 秒死)→ apt 装 libgl1
10. numpy 1.26.4 vs pyarrow 26 → 升回 2.2.6(三方共存验证过)
11. ray job 工作目录不在 slime → 脚本加 `cd ${SLIME_DIR}`
12. **8 卡机误用 16 卡模板**(actor 8+rollout 8=16 卡需求→placement 死锁)→ 改官方 colocate 配方:`--colocate`+TP4+`TMS_HOOK_MODE=preload`+`ASCEND_COREDUMP_SIGNAL=none`+禁 expandable_segments
13. **GDN 状态池 FP32 仅 Ascend950 支持**(910B3 算子只收 BF16)→ `SGLANG_MAMBA_SSM_DTYPE=bfloat16`
14. dapo 数据无顶层 label 字段 → fix_dapo_label.py 补齐(deepscaler RM 要 str)

## 正式跑之前必须解决

- **磁盘**:每个 ckpt ≈50G(权重+优化器),正式档每 20 轮存一次,现有 14G 远不够 → 需要拍板:清理旧实验数据(qsa-indexer 127G 等)或换 ckpt 存放路径(脚本里 `--save` 可改 /mnt 等)
- 评测数据 aime-2024 机器上没有(EVAL_ARGS 已注释,链路通了可补)

## 沙箱(sunabako)说明

slime 的 Rust 沙箱运行时 sunabako 源码只支持 x86_64,aarch64 编译不过,已装明确报错的占位包。本任务的沙箱 = slime/examples/retool/tool_sandbox.py(纯 Python:subprocess 隔离+黑白名单+限额,不依赖 sunabako)。若将来要跑 coding_agent_rl(SWE 沙箱 RL),需要独立移植 sunabako 到 aarch64(机器上有现成 aarch64 rust 1.100 工具链)。

---

# retool 沙箱冒烟(2026-11-20,方向 A:本地 tool_sandbox.py)

## 结论

**通过**。Qwen3.5-9B 的工具调用 RL 全链路(多轮生成→本地沙箱执行→结果回灌→math_dapo+工具塑形奖励→GRPO 训练)在 8×910B3 上端到端跑通 2 轮,ray job 官方判定 **succeeded(rc=0,干净退出,连最终 ckpt 都存了 iter_0000001,172G)**。

## 接入方式(在已验证的 GRPO 底座上加四件事)

1. 脚本 `run-qwen3.5-9B-npu-retool.sh`(本目录 scripts/,w13 同份 md5 f0a9a1de):
   - `--custom-generate-function-path generate_with_retool.generate`(多轮工具循环)
   - `--custom-rm-path generate_with_retool.reward_func`(自定义奖励)
   - PYTHONPATH 加 `slime/examples/retool`(驱动 export + runtime-env 双份)
   - 数据换 `dapo-retool.jsonl`(prompt 拍平成纯字符串+label,fix_dapo_retool.py 生成)
2. 沙箱本体零改动:`examples/retool/tool_sandbox.py` 纯 Python,容器原生可跑。

## 沙箱执行证据(日志 grep 复核)

- **27 次 `<interpreter>` 沙箱执行回灌**;sandbox 拒绝 `import sympy`、拦截 `__import__`、执行带 print 的代码返回真实输出("Output:\n31000")
- 模型真实调工具:解 gcd 求和题直接写 Python 交给 code_interpreter 并拿到 31000;换策略绕 import 限制(import math 成功)
- 2 轮 rollout 各 32 样本,训练步完成,权重同步 v1→v2→v3(update_weights_from_tensor ×117)
- 本轮样本多轮上下文均值 5057 token(工具模板+多轮回灌),rollout 496 秒/轮

## 已知观察(均非管线 bug,RL 冷启动现象)

- 原模型没学过工具格式:会出现 code 传成 list(被纠正)、answer 忘 \boxed、重复同一调用——正是训练要收敛的东西
- response 全截断率 ~47-59%(多轮吃上下文比纯文本快)
- recomp 段(8 样本)扣 0.1 是上游预置策略

## 重跑

```
docker exec czl_slime bash -lc 'SMOKE_MODE=1 bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu-retool.sh'
```
沙箱单测(不占卡):`python3 scripts/test_tool_sandbox.py`(三段:计算/危险 import/死循环超时,已全过)

## 正式跑提示

- ckpt 172G 已存 1 份,磁盘剩 59G——正式跑(3000 轮×每 20 轮存)前必须清旧 ckpt或改 --save 路径
- 磁盘充足后直接去掉 SMOKE_MODE=1 即正式档
