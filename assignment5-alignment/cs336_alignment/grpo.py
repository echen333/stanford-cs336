import torch
from typing import Callable, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = torch.tensor(
        [
            reward_fn(response, ground_truth)["reward"]
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
        ]
    ).view(-1, group_size)

    print(raw_rewards.shape)
    advantages = raw_rewards - raw_rewards.mean(dim=-1).unsqueeze(-1)

    if normalize_by_std:
        stds = advantages.std(dim=-1).unsqueeze(-1)
        advantages = advantages / (stds + advantage_eps)

    raw_rewards = raw_rewards.view(-1)
    advantages = advantages.view(-1)
    metadata = {}
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = {"max": 1.0}
    ratio = torch.exp(policy_log_probs - old_log_probs)

    loss = -torch.minimum(
        ratio * advantages,
        torch.clip(ratio, 1 - cliprange, 1 + cliprange) * advantages,
    )
    return loss, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    # breakpoint()
    tensor_sum = torch.masked_fill(tensor, ~mask, 0)
    return torch.sum(tensor_sum, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(token_loss, response_mask) / (gradient_accumulation_steps)
    loss.backward()
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    else:
        assert False, "loss type is not valid"
