from torch import Tensor
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    assert len(prompt_strs) == len(output_strs)
    encoded_prompts = tokenizer(prompt_strs)
    encoded_outputs = tokenizer(output_strs)

    max_len = 0
    for x, y in zip(encoded_prompts["input_ids"], encoded_outputs["input_ids"]):
        max_len = max(max_len, len(x) + len(y))

    encoded = []
    for x, y in zip(encoded_prompts["input_ids"], encoded_outputs["input_ids"]):
        encoded.append(x + y + [tokenizer.eos_token_id] * (max_len - len(x) - len(y)))

    encoded = torch.tensor(encoded)
    mask = torch.zeros_like(encoded, dtype=torch.bool)
    for i, (p, q) in enumerate(
        zip(encoded_prompts["input_ids"], encoded_outputs["input_ids"])
    ):
        mask[i, len(p) : len(p) + len(q)] = 1

    return {
        "input_ids": encoded[:, :-1],
        "labels": encoded[:, 1:],
        "response_mask": mask[:, 1:],
    }


def compute_entropy(logits: Tensor):
    assert logits.ndim == 3
    sub_max = logits - logits.max(-1, keepdim=True)[0]
    exp_sub = torch.exp(sub_max)
    p = exp_sub / torch.sum(exp_sub, -1).unsqueeze(-1)
    ans = torch.sum(p * torch.log(p), -1)
    return -ans


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    logits = model(input_ids)["logits"]

    B, C, V = logits.shape
    ret = {}
    if return_token_entropy:
        ret["token_entropy"] = compute_entropy(logits)

    logits = logits.flatten(0, 1)
    labels = labels.flatten()
    logits = F.log_softmax(logits)
    log_probs: Tensor = logits[torch.arange(labels.numel()), labels]
    log_probs = log_probs.unflatten(0, (B, C))
    ret["log_probs"] = log_probs

    return ret


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim) / normalize_constant


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    print(gradient_accumulation_steps, normalize_constant)
    B = policy_log_probs.shape[0]
    print("B", B)
    loss = (
        -masked_normalize(policy_log_probs, response_mask, None, B * normalize_constant)
        / gradient_accumulation_steps
    )
    loss.backward()

    metadata = {"entropy": 1.0}
    print(f"loss {loss}")
    print(metadata)
    return loss, metadata


def log_generations(model, prompt):
    pass
