#!/usr/bin/env bash
# ---------------------------------------------
# run_multi_gpu.sh  —  并行在多 GPU 上批量推理
# 用法：
#   ./run_multi_gpu.sh [--date 20250512] [--root output_klora_ipadapter]
# ---------------------------------------------
set -euo pipefail

############### 1. 可选 CLI 参数 ################
DATE=$(date +%Y%m%d)            # 默认当天
ROOT="output_klora_ipadapter"   # 输出根目录
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date) DATE="$2"; shift 2;;
    --root) ROOT="$2"; shift 2;;
    *) echo "未知参数 $1"; exit 1;;
  esac
done

############### 2. 任务维度 #####################
# 只需要在这里维护名字 ➜ 索引的映射
CONTENT=(lip anna richy)        # content_index = 下标
STYLE=(3d boldeasy bwphoto crosshatch sportsposter \
       inkysketch linkedin silhouette vintagephoto greenTint)  # style_index = 下标

############### 3. GPU 配置 #####################
GPU_LIST=(1 2 3 4)          # 想用哪些 GPU 就写哪些
NGPU=${#GPU_LIST[@]}

############### 4. 生成并发任务 ##################
run_one() {
  local c_idx=$1 s_idx=$2
  local c_name=${CONTENT[$c_idx]}
  local s_name=${STYLE[$s_idx]}
  local gpu_id=${GPU_LIST[$(( (c_idx*${#STYLE[@]}+s_idx) % NGPU ))]}

  local out_folder=./${ROOT}_${DATE}/output_${c_name}_${s_name}
  local cmd="python inference_flux_with_seekoo_iplora.py \
      --output_folder ${out_folder} \
      --content_index ${c_idx} \
      --style_index   ${s_idx}"

  echo "[GPU ${gpu_id}] $cmd"
  (
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    eval "$cmd"
  ) &
}

############### 5. 启动 #########################
for c_idx in "${!CONTENT[@]}"; do
  for s_idx in "${!STYLE[@]}";  do
    run_one "$c_idx" "$s_idx"
    # 控制最大并发 = NGPU
    if (( $(jobs -r | wc -l) >= NGPU )); then
      wait -n   # Bash 5+: 等待任意任务结束
    fi
  done
done
wait
echo "✅ 全部任务已完成 —— $(date)."