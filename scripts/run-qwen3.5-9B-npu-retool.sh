#!/bin/bash
# 冒烟: docker exec czl_slime bash -lc 'SMOKE_MODE=1 bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu-retool.sh'
# 正式: docker exec czl_slime bash -lc 'bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu-retool.sh'

# 复跑前清场
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
SLIME_DIR=/home/c00954327/slime-ascend/slime
MEGATRON_DIR=/home/c00954327/Megatron-LM
BRIDGE_DIR=/home/c00954327/Megatron-Bridge
MBRIDGE_DIR=/home/c00954327/mbridge
SGLANG_DIR=/home/c00954327/sglang
RETOOL_DIR=${SLIME_DIR}/examples/retool

HF_CKPT=/home/zrw/Qwen3.5-9B                                   # 权重
PROMPT_DATA_SRC=/home/zhouyang/datasets/slime_dapo/dapo-math-17k.jsonl # 数据集

# 运行产物目录(转换权重/训练 ckpt/日志)
BASE=/home/c00954327/slime-run
TORCH_DIST=${BASE}/Qwen3.5-9B_torch_dist                        # 已缓存的转换产物
SLIME_CKPT=${BASE}/Qwen3.5-9B_slime
mkdir -p "${BASE}/logs" "${BASE}/data"
# retool 数据:prompt 拍平成纯字符串 + label(generate_with_retool 的模板要字符串)
PROMPT_DATA=${BASE}/data/dapo-retool.jsonl
if [ ! -f "${PROMPT_DATA}" ]; then
    python3 "${BASE}/data/fix_dapo_retool.py"
fi

# ---------- 冒烟 / 正式 ----------
# 跑几轮，每轮几题，每步训练的样本总数 = 题数 × 8（每题采8条），单轮生成的token上限，每N步存ckpt
SMOKE_MODE=${SMOKE_MODE:-0}
if [ "${SMOKE_MODE}" = "1" ]; then
    NUM_ROLLOUT=2; ROLLOUT_BS=4; GBS=32; RESP_LEN=2048; SAVE_INTERVAL=1000
else
    # 可通过环境变量覆盖(如 NUM_ROLLOUT=100 SAVE_INTERVAL=100 bash …)
    NUM_ROLLOUT=${NUM_ROLLOUT:-3000}; ROLLOUT_BS=${ROLLOUT_BS:-32}; GBS=${GBS:-256}; RESP_LEN=${RESP_LEN:-4096}; SAVE_INTERVAL=${SAVE_INTERVAL:-20}
fi

# ---------- 前置检查 ----------
for p in "${SLIME_DIR}/train.py" "${MEGATRON_DIR}" "${SGLANG_DIR}/python" \
         "${BRIDGE_DIR}" "${MBRIDGE_DIR}" "${RETOOL_DIR}/generate_with_retool.py" \
         "${RETOOL_DIR}/tool_sandbox.py" "${HF_CKPT}" "${PROMPT_DATA}"; do
    if [ ! -e "${p}" ]; then
        echo "[ERROR] 缺少 ${p} —— 依赖未装全或数据未就位"
        exit 1
    fi
done

# ---------- NPU 环境变量 ----------
export PYTHONPATH="${MEGATRON_DIR}:${BRIDGE_DIR}/src:${MBRIDGE_DIR}:${SGLANG_DIR}/python:${SLIME_DIR}:${RETOOL_DIR}:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
# colocate 模式必备(官方 colocate 模板 Notes 3/4):TMS preload 钩子 + 关 CANN 核心转储信号
export TMS_HOOK_MODE="preload"
export ASCEND_COREDUMP_SIGNAL=none
# 910B3 的 npu chunk_gated_delta_rule 算子只收 BF16 initial_state(FP32 仅 950/arch35 支持)
export SGLANG_MAMBA_SSM_DTYPE="bfloat16"
# 注意:colocate 模式禁用 expandable_segments(官方 Notes 6:会让 offload 失效导致 OOM)

# 模型结构参数(上游官方 qwen3.5-9B 配置)
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"

# ---------- 权重转换(HF → Megatron torch_dist,已缓存则跳过)，重跑时可自动跳过转换 ----------
if [ ! -d "${TORCH_DIST}" ]; then
    echo "[INFO] 未找到 torch_dist,开始转换(约 19G)……"
    (cd "${SLIME_DIR}" && python3 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_CKPT}" \
        --save "${TORCH_DIST}")
fi

# ray job 的工作目录 = ray head 启动时的 cwd,必须先进 slime 目录
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
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BS}
   --n-samples-per-prompt 8
   --rollout-max-response-len ${RESP_LEN}
   --rollout-temperature ${ROLLOUT_TEMP:-1.0}
   --global-batch-size ${GBS}
   --balance-data
)
# 注:不传 --rm-type,奖励走 CUSTOM_ARGS 的自定义 RM(generate_with_retool.reward_func)

# 评测(aime-2024 数据未下载)，暂时跳过
EVAL_ARGS=()

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
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
   --no-gradient-accumulation-fusion
)
# 注:max-tokens-per-gpu 决定多轮样本的总上下文预算(CP×该值),也是训练微批的
# token 上限;可用 MAX_TOKENS_PER_GPU=20480 环境变量放大(8B 官方模板同值)。

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
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

# retool 核心接线(官方 examples/retool 用法):自定义生成 + 自定义奖励
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

export LOG_FILE="${BASE}/logs/train_log_$(date +%Y-%m-%d_%H-%M-%S)_qwen3.5_9b_retool.log"

# ray head(容器内单机)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_DEBUG=legacy \
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external

# ray job 运行时环境(容器内真实 CANN 路径;PYTHONPATH 含 retool 例子目录)
TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${BRIDGE_DIR}/src:${MBRIDGE_DIR}:${SGLANG_DIR}/python:${SLIME_DIR}:${RETOOL_DIR}\",
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
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
