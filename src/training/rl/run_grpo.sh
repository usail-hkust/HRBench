#!/usr/bin/env bash
# ============================================================================
# GRPO Training Script for Hybrid Reasoning
# Uses verl's PPO trainer with GRPO advantage estimator
#
# Reward: α × accuracy + β × efficiency × accuracy  (correct & concise)
#         -1.0 for wrong answers
#
# Usage:
#   # Default: Qwen3.5-2B, math500 prompts
#   bash src/training/rl/run_grpo.sh
#
#   # From SFT checkpoint:
#   MODEL_PATH=checkpoints/sft/qwen3.5-2b_sft_math500/... \
#   bash src/training/rl/run_grpo.sh
# ============================================================================
set -x

# ==================== Paths ====================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ==================== Environment ====================
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Model — base model or SFT checkpoint
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3.5-2B"}

# Data — GRPO prompts from build_training_data.py
TRAIN_PATH=${TRAIN_PATH:-"${PROJECT_ROOT}/results/training_data/qwen3.5-2b/math500/grpo_train.parquet"}
TEST_PATH=${TEST_PATH:-"${TRAIN_PATH}"}  # reuse train as test for now

# Save
EXP_NAME=${EXP_NAME:-"qwen3.5-2b_grpo_math500"}
SAVE_PATH=${SAVE_PATH:-"${PROJECT_ROOT}/checkpoints/rl/${EXP_NAME}"}
mkdir -p "${SAVE_PATH}"

# Reward function
REWARD_PATH="${PROJECT_ROOT}/src/training/rl/reward_function.py"

# ==================== Launch ====================
echo "============================================="
echo "  Hybrid Reasoning GRPO Training"
echo "  Model: ${MODEL_PATH}"
echo "  Data:  ${TRAIN_PATH}"
echo "  Save:  ${SAVE_PATH}"
echo "============================================="

HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PATH}" \
    data.val_files="${TEST_PATH}" \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    +data.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.alpha=1.0 \
    +reward.custom_reward_function.reward_kwargs.beta=0.5 \
    +reward.custom_reward_function.reward_kwargs.max_tokens=8192 \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name="HybridReasoningRL" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${SAVE_PATH}" \
    "$@"

echo "GRPO training complete. Checkpoints at: ${SAVE_PATH}"
