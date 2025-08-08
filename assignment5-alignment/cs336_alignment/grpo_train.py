import torch
from grpo import grpo_microbatch_train_step, compute_group_normalized_rewards
from torch.optim import AdamW
import wandb
from vllm import SamplingParams, LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cs336_alignment.math_baseline import evaluate_vllm, get_prompt, get_prompt_template
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    run_sft_microbatch_train_step,
    log_generations,
    get_response_log_probs,
    run_tokenize_prompt_and_output,
    load_policy_into_vllm_instance,
    init_vllm,
)


def convert_log_probs(log_probs:list[list[float]]):
    max_len = max(len(seq) for seq in log_probs)
    log_probs_tensor = torch.zeros(len(log_probs), max_len)
    response_mask = torch.zeros(len(log_probs), max_len, dtype=torch.bool)

    for i, seq in enumerate(log_probs):
        length = len(seq)
        log_probs_tensor[i, :length] = torch.Tensor(seq)
        response_mask[i, :length] = 1
    return log_probs_tensor, response_mask

def run_generation_worker(output_dir: str, prompts, sampling_params):
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    llm: LLM = init_vllm(output_dir, "cuda:1", 42)
    load_policy_into_vllm_instance(model, llm)

    outputs = llm.generate(prompts, sampling_params)
    return outputs


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
        "device": "cuda:0",
        "seed": 42,
    }
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=cfg["sampling_max_tokens"],
        stop=["\n"],
        n=cfg["group_size"],
        seed=cfg["seed"],
        min_tokens=cfg["sampling_min_tokens"],
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    sampling_params.logprobs=1

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    dataset = dataset.shuffle(seed=cfg["seed"])

    run = wandb.init(entity="eddys", project="stanford-lm-5", config=cfg, group="grpo")
    micro_train_batch_size = (
        cfg["train_batch_size"] // cfg["gradient_accumulation_steps"]
    )
    assert cfg["rollout_batch_size"] % cfg["group_size"] == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = cfg["rollout_batch_size"] // cfg["group_size"]
    assert cfg["train_batch_size"] >= cfg["group_size"], (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = (
        cfg["rollout_batch_size"] // micro_train_batch_size
    )
    generate_process = None
    for step in range(1, cfg["n_grpo_steps"] + 1):
        output_dir = "data/echen333/grpo_policy"
        model.save_pretrained(save_directory=output_dir)
        tokenizer.save_pretrained(save_directory=output_dir)

        dataset.shuffle(step)

        data = dataset[:n_prompts_per_rollout_batch]
        prompt_strs = dataset["problem"][:n_prompts_per_rollout_batch]
        solutions = dataset["solution"][:n_prompts_per_rollout_batch]
        prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
        prompts = [get_prompt(prompt_template, x) for x in prompt_strs]
        responses = run_generation_worker(output_dir, prompts, sampling_params)

        # old_logits = get_response_log_probs(model, all_input_ids, all_labels, False)
        old_logits = None
        completions = [x for response in responses for x in response.outputs] # : list[CompletionOutput]
        policy_log_probs = [[log_prob[id].logprob for log_prob, id, in zip(completion.logprobs, completion.token_ids)] for completion in completions ]
        policy_log_probs, response_mask = convert_log_probs(policy_log_probs)

        # tokenized = run_tokenize_prompt_and_output(prompts, ground_truth, tokenizer)
        # all_input_ids = tokenized["input_ids"].to(device)
        # all_labels = tokenized["labels"].to(device)
        # response_mask = tokenized["response_mask"].to(device)
        # policy_log_probs = 

        repeated_ground_truths = [solution for solution in solutions for _ in range(cfg["group_size"])]

        # need to get old log probs
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            [x.text for x in completions],
            repeated_ground_truths,
            cfg["group_size"],
            cfg["advantage_eps"],
            cfg["use_std_normalization"],
        )
        advantages = advantages.unsqueeze(-1)
        raw_rewards = raw_rewards.unsqueeze(-1)
        breakpoint()
        for train_step in range(1, cfg["epochs_per_rollout_batch"] + 1):
            loss, metadata2 = grpo_microbatch_train_step(policy_log_probs,response_mask,cfg["gradient_accumulation_steps"],cfg["loss_type"],raw_rewards,advantages,old_logits,None,)
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    main()
