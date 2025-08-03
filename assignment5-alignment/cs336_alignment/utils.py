from torch import Tensor
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


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
    normalized = F.log_softmax(logits, -1)
    ans = torch.sum(torch.exp(normalized) * normalized, -1)
    return -ans


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids)["logits"]

    B, C, V = logits.shape
    ret = {}
    if return_token_entropy:
        ret["token_entropy"] = compute_entropy(logits)

    logits = logits.flatten(0, 1)
    labels = labels.flatten()

    log_probs = -F.cross_entropy(logits, labels, reduction='none')
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
    B = policy_log_probs.shape[0]
    loss = (
        -masked_normalize(policy_log_probs, response_mask, None, B * normalize_constant)
        / gradient_accumulation_steps
    )
    loss.backward()

    metadata = {"entropy": 1.0}
    return loss, metadata


def log_generations(model, prompt):
    pass


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
