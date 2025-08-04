import torch
from grpo import grpo_microbatch_train_step, compute_group_normalized_rewards
from torch.optim import AdamW


def main():
    cfg = {
        "n_grpo_steps": 200,
        "learning_rate": 1e-5,
        "advantage_eps": 1e-6,
        "rollout_batch_size": 256,
        "group_size": 8,
        "sampling_temperature": 1.0,
        "sampling_min_tokens": 4,  # As in Expiter, disallow empty string responses
        "sampling_max_tokens": 1024,
        "epochs_per_rollout_batch": 1,  # On-policy
        "train_batch_size": 256,  # On-policy
        "gradient_accumulation_steps": 128,  # microbatch size is 2, will fit on H100
        "gpu_memory_utilization": 0.85,
        "loss_type": "reinforce_with_baseline",
        "lr": 2e-4,
        "use_std_normalization": True,
    }
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg["lr"],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # assert train_batch_size % cfg["gradient_accumulation_steps"] == 0, (
    # "train_batch_size must be divisible by gradient_accumulation_steps"
    # )
    micro_train_batch_size = train_batch_size // gradient_accumulation_stepsassert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_sizeassert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size


if __name__ == "__main__":
    main()
