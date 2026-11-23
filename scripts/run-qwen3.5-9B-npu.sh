#!/bin/bash
# ============================================================================
# run-qwen3.5-9B-npu.sh — Qwen3.5-9B GRPO 训练(8×910B3 单机,czl_slime 容器内)
#
# 改写自 scripts/run-qwen3-8B-npu.sh(NPU 已验证模板);
# 模型结构参数 source 自 slime/scripts/models/qwen3.5-9B.sh(上游官方配置)。
#
# 前置条件(缺一不可,不齐会启动失败):
#   1. 容器内按 scripts/quick_install.sh 装好依赖:
#      /home/c00954327/{Megatron-LM, sglang, Megatron-Bridge, mbridge,
#      MegatronAdaptor, TransformerEngineNPU},slime 已 pip install -e .
#      注意:quick_install 多处要从 github 克隆/下载,w13 直连 github 不通,
#      需要"本机克隆→tar 管道"中转。
#   2. HF 权重:/home/zrw/Qwen3.5-9B(19G BF16,只读引用,不改动)。
#   3. 训练数据 dapo-math-17k.jsonl:脚本自动从周洋目录拷一份(1.3G,一次性)。
#
# 用法(w13 上,容器内执行):
#   冒烟(先跑这个): docker exec czl_slime bash -lc 'SMOKE_MODE=1 bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu.sh'
#   正式:           docker exec czl_slime bash -lc 'bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu.sh'
#
# 已知风险(按顺序排查):
#   A. 权重转换:Qwen3.5-9B 是多模态版(Qwen3_5ForConditionalGeneration),
#      而转换走纯文本 spec(get_qwen3_5_spec);若 convert 报不认识架构,
#      要查 mbridge/Megatron-Bridge 对 qwen3_5 VL 检查点的处理。
#   B. 9B 未进 NPU 官方支持表(只验证过 8B/30B-A3B/GLM-4.7-Flash),
#      算子与显存问题要有预期,冒烟先探路。
#   C. /home 仅剩 ~78G:torch_dist 约 19G + 每次训练 ckpt 约 19G+,
#      save-interval 别调太小,跑完及时清旧 ckpt。
# ============================================================================

# 复跑前清理(沿用官方模板做法)
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex
ulimit -n 65535
export PYTHONUNBUFFERED=1

# ---------- 路径(与 quick_install.sh 的安装布局一致) ----------
SLIME_ASCEND_ROOT=/home/c00954327/slime-ascend
SLIME_DIR=${SLIME_ASCEND_ROOT}/slime
MEGATRON_DIR=/home/c00954327/Megatron-LM
BRIDGE_DIR=/home/c00954327/Megatron-Bridge
MBRIDGE_DIR=/home/c00954327/mbridge
SGLANG_DIR=/home/c00954327/sglang

HF_CKPT=/home/zrw/Qwen3.5-9B                                   # HF 权重(只读)
PROMPT_DATA_SRC=/home/zhouyang/datasets/slime_dapo/dapo-math-17k.jsonl

# 运行产物目录(转换权重/训练 ckpt/日志)
BASE=/home/c00954327/slime-run
TORCH_DIST=${BASE}/Qwen3.5-9B_torch_dist                        # HF→Megatron 转换产物,约 19G
SLIME_CKPT=${BASE}/Qwen3.5-9B_slime                             # 训练 load/save 目录
mkdir -p "${BASE}/logs" "${BASE}/data"
# 数据:verl 格式源(无顶层 label)→ 转换出带 label 的版本(deepscaler RM 需要 str 型 label)
PROMPT_DATA=${BASE}/data/dapo-math-17k-label.jsonl
if [ ! -f "${PROMPT_DATA}" ]; then
    if [ ! -f "${BASE}/data/dapo-math-17k.jsonl" ] && [ -f "${PROMPT_DATA_SRC}" ]; then
        cp "${PROMPT_DATA_SRC}" "${BASE}/data/dapo-math-17k.jsonl"
    fi
    python3 "${BASE}/data/fix_dapo_label.py"
fi

# ---------- 冒烟 / 正式 两档参数 ----------
SMOKE_MODE=${SMOKE_MODE:-0}
if [ "${SMOKE_MODE}" = "1" ]; then
    # 冒烟:2 轮 rollout、小 batch、短回复、不存 ckpt —— 只验证链路能转
    NUM_ROLLOUT=2; ROLLOUT_BS=4; GBS=32; RESP_LEN=2048; SAVE_INTERVAL=1000
else
    NUM_ROLLOUT=3000; ROLLOUT_BS=32; GBS=256; RESP_LEN=4096; SAVE_INTERVAL=20
fi

# ---------- 前置检查(缺什么提前报,别等 ray 起来才炸) ----------
for p in "${SLIME_DIR}/train.py" "${MEGATRON_DIR}" "${SGLANG_DIR}/python" \
         "${BRIDGE_DIR}" "${MBRIDGE_DIR}" "${HF_CKPT}" "${PROMPT_DATA}"; do
    if [ ! -e "${p}" ]; then
        echo "[ERROR] 缺少 ${p} —— 依赖未装全(先按 scripts/quick_install.sh 装)或数据未就位"
        exit 1
    fi
done
echo "[INFO] HF 权重架构: $(grep -m1 '"architectures"' -A1 ${HF_CKPT}/config.json)"

# ---------- NPU 环境变量(沿用官方模板) ----------
export PYTHONPATH="${MEGATRON_DIR}:${BRIDGE_DIR}/src:${MBRIDGE_DIR}:${SGLANG_DIR}/python:${SLIME_DIR}:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
# colocate 模式必备(官方 colocate 模板 Notes 3/4):TMS preload 钩子 + 关 CANN 核心转储信号
export TMS_HOOK_MODE="preload"
export ASCEND_COREDUMP_SIGNAL=none
# 910B3 的 npu chunk_gated_delta_rule 算子只收 BF16 initial_state(FP32 仅 950/arch35 支持),
# GDN 状态池默认 float32 会直接 RuntimeError,这里用官方环境变量盖掉
export SGLANG_MAMBA_SSM_DTYPE="bfloat16"
# 注意:colocate 模式禁用 expandable_segments(官方 Notes 6:会让 offload 失效导致 OOM)

# 模型结构参数(上游官方 qwen3.5-9B 配置:16 头 / 4 KV 组 / head_dim 256 /
# 32 层 / hidden 4096 / vocab 248320 / rope 1e7 / 25% rotary / gated attention)
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"

# ---------- 权重转换(HF → Megatron torch_dist,只转一次) ----------
if [ ! -d "${TORCH_DIST}" ]; then
    echo "[INFO] 未找到 torch_dist,开始转换(约 19G)……"
    (cd "${SLIME_DIR}" && python3 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_CKPT}" \
        --save "${TORCH_DIST}")
fi

# ray job 的 entrypoint(python3 train.py)工作目录 = ray head 启动时的 cwd,
# 必须先进 slime 目录,否则报 can't open file '/…/train.py'
cd "${SLIME_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${TORCH_DIST}/"
   --load "${SLIME_CKPT}/"
   --save "${SLIME_CKPT}/"
   --save-interval ${SAVE_INTERVAL}
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BS}
   --n-samples-per-prompt 8
   --rollout-max-response-len ${RESP_LEN}
   --rollout-temperature 0.8
   --global-batch-size ${GBS}
   --balance-data
)

# 评测(aime-2024 数据未下载;链路打通后再补数据并解开)
EVAL_ARGS=()
# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data aime /home/c00954327/slime-run/data/aime-2024.jsonl
#    --n-samples-per-eval-prompt 16
#    --eval-max-response-len 16384
#    --eval-top-p 0.7
# )

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --no-gradient-accumulation-fusion
)
# 注:8B 模板 max-tokens-per-gpu 20480;9B 的 head_dim 256 更吃显存,先保守取 16384,
# 稳定后可往上加。

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 4 8 16 32 64 128
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 32768
   --sglang-max-prefill-tokens 4000
   --sglang-max-total-tokens 327680
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
   # Qwen3.5-9B 带 1 层 MTP,链路通了以后可试 27B 脚本的 EAGLE 投机解码:
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

export LOG_FILE="${BASE}/logs/train_log_$(date +%Y-%m-%d_%H-%M-%S)_qwen3.5_9b.log"

# ray head(容器内单机)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_DEBUG=legacy \
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external

# ray job 运行时环境(容器内真实 CANN 路径)
TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${BRIDGE_DIR}/src:${MBRIDGE_DIR}:${SGLANG_DIR}/python:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"ASCEND_TOOLKIT_HOME\": \"${TOOLKIT_HOME}\",
    \"ASCEND_OPP_PATH\": \"${TOOLKIT_HOME}/opp/\",
    \"ASCEND_AICPU_PATH\": \"${TOOLKIT_HOME}\",
    \"ASCEND_HOME_PATH\": \"${TOOLKIT_HOME}\",
    \"set_env_path\": \"/usr/local/Ascend/nnal/atb/set_env.sh\",
    \"TMS_HOOK_MODE\": \"preload\",
    \"ASCEND_COREDUMP_SIGNAL\": \"none\",
    \"SGLANG_MAMBA_SSM_DTYPE\": \"bfloat16\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"RAY_DEBUG_POST_MORTEM\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
