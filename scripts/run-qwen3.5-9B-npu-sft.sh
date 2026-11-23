#!/bin/bash
# ============================================================================
# run-qwen3.5-9B-npu-sft.sh — Qwen3.5-9B retool SFT 冷启动
# (官方 examples/retool/retool_qwen3_4b_sft.sh 的 910B/Qwen3.5-9B 适配版)
#
# 官方 retool 配方是两阶段:SFT 冷启动(本脚本,教模型工具调用格式)
#   → RL(run-qwen3.5-9B-npu-retool.sh,练解题拿分)。
# 此前我们直接从基座进 RL,前 11 步 raw_reward 才转正的冷启动现象即源于
# 跳过了本阶段。
#
# 与 RL 版的差异:
#   1. 纯监督训练:无 SGLang 引擎/无沙箱/无奖励(--debug-train-only),
#      不用 colocate,因此 TMS preload / MAMBA dtype 那套环境变量也不需要;
#   2. loss=sft_loss(只学 assistant token),无 GRPO/优势/自定义 generate;
#   3. 数据=ReTool-SFT.parquet(首次运行经 hf-mirror 自动下载并转换);
#   4. lr 1e-5 cosine 衰减(官方 SFT 配方,约 10 倍于 RL 学习率);
#   5. --no-save-optim:产物只用于喂 RL,不需要优化器状态,单份存盘
#      从 ~172G 降到几十 G(run100 零 ckpt 教训后可多存几份)。
#      注意:该开关本栈首次使用,第一次存盘时盯一眼日志存盘行。
#
# 用法:
#   冒烟: docker exec czl_slime bash -lc 'SMOKE_MODE=1 bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu-sft.sh'
#   正式: docker exec czl_slime bash -lc 'bash /home/c00954327/slime-ascend/scripts/run-qwen3.5-9B-npu-sft.sh'
#
# 注意:SFT 没有 num-rollout 式步数上限,冒烟档(1 epoch)看前几十步
# loss 正常下降即可手动停;正式档 3 epoch 跑满。
# ============================================================================

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

# ---------- 路径(与 quick_install 布局一致) ----------
SLIME_DIR=/home/c00954327/slime-ascend/slime
MEGATRON_DIR=/home/c00954327/Megatron-LM
BRIDGE_DIR=/home/c00954327/Megatron-Bridge
MBRIDGE_DIR=/home/c00954327/mbridge
SGLANG_DIR=/home/c00954327/sglang

HF_CKPT=/home/zrw/Qwen3.5-9B                                   # 基座权重(只读)

BASE=/home/c00954327/slime-run
TORCH_DIST=${BASE}/Qwen3.5-9B_torch_dist                        # 与 RL 共用的转换缓存,守卫跳过
SFT_CKPT=${BASE}/Qwen3.5-9B_sft_slime                           # SFT 产物(RL 冷启动权重)
mkdir -p "${BASE}/logs" "${BASE}/data"
# 全程落日志:从本行起所有输出(含预检/转换)都 tee 进 LOG_FILE,
# 早期失败也能留下日志(旧版 tee 只挂在 ray job submit,早期阶段不落盘)
export LOG_FILE="${BASE}/logs/train_log_$(date +%Y-%m-%d_%H-%M-%S)_qwen3.5_9b_sft.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
PROMPT_DATA=${BASE}/data/ReTool-SFT.parquet
if [ ! -f "${PROMPT_DATA}" ]; then
    python3 "${BASE}/data/fix_retool_sft_data.py"               # 首次会经 hf-mirror 下载 JoeYing/ReTool-SFT
fi

# ---------- 冒烟 / 正式 ----------
SMOKE_MODE=${SMOKE_MODE:-0}
if [ "${SMOKE_MODE}" = "1" ]; then
    NUM_EPOCH=1; ROLLOUT_BS=8; GBS=8; SAVE_INTERVAL=1000
else
    NUM_EPOCH=${NUM_EPOCH:-3}; ROLLOUT_BS=${ROLLOUT_BS:-128}; GBS=${GBS:-128}; SAVE_INTERVAL=${SAVE_INTERVAL:-100}
fi

# ---------- 前置检查(注意:torch_dist 由下方转换守卫创建,不在此预检) ----------
for p in "${SLIME_DIR}/train.py" "${MEGATRON_DIR}" "${BRIDGE_DIR}" \
         "${MBRIDGE_DIR}" "${SGLANG_DIR}/python" "${HF_CKPT}" \
         "${PROMPT_DATA}"; do
    if [ ! -e "${p}" ]; then
        echo "[ERROR] 缺少 ${p} —— 依赖未装全或数据未就位"
        exit 1
    fi
done

# ---------- NPU 环境变量(纯训练,无 rollout:TMS/colocate/MAMBA 均不需要) ----------
export PYTHONPATH="${MEGATRON_DIR}:${BRIDGE_DIR}/src:${MBRIDGE_DIR}:${SGLANG_DIR}/python:${SLIME_DIR}:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export ASCEND_COREDUMP_SIGNAL=none

# 模型结构参数(与 RL 同源,同一份转换缓存)
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"

# ---------- 权重转换(HF → Megatron torch_dist,与 RL 共用缓存,只转一次) ----------
if [ ! -d "${TORCH_DIST}" ]; then
    echo "[INFO] 未找到 torch_dist,开始转换(约 19G,一次性)……"
    (cd "${SLIME_DIR}" && python3 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_CKPT}" \
        --save "${TORCH_DIST}")
fi

# ray job 工作目录
cd "${SLIME_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${TORCH_DIST}/"
   --load "${SFT_CKPT}/"
   --save "${SFT_CKPT}/"
   --save-interval ${SAVE_INTERVAL}
   --no-save-optim
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch ${NUM_EPOCH}
   --rollout-batch-size ${ROLLOUT_BS}
   --global-batch-size ${GBS}
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# 官方 4B 用 TP1+CP2;9B 沿用已验证的 TP4+SP 配方,长对话 OOM 时加 --context-parallel-size 2
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
   --log-probs-chunk-size 1024
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
   --no-gradient-accumulation-fusion
)

# 官方 SFT 优化器配方:lr 1e-5 cosine→1e-6,10% 预热,beta2 0.95
OPTIMIZER_ARGS=(
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=()
EVAL_ARGS=()

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

# ray head(容器内单机)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# ray job 运行时环境(纯训练,CANN 路径保留;无 TMS/MAMBA)
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
    \"ASCEND_COREDUMP_SIGNAL\": \"none\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"RAY_DEBUG_POST_MORTEM\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
# 注:输出已由脚本头部的 exec tee 全程落 LOG_FILE;此处不再接管道,
# 退出码也更真实(ray job submit 本身的 rc,而非 tee 的恒 0)
