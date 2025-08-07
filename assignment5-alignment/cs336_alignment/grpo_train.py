import torch
from grpo import grpo_microbatch_train_step, compute_group_normalized_rewards
from torch.optim import AdamW
import wandb
from vllm import SamplingParams, LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def 
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

    sampling_min_tokens = 4
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["\n"],
        n=cfg["G"],
        seed=cfg["seed"],
        min_tokens=sampling_min_tokens,
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

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
    for step in range(1, cfg["n_grpo_steps"] + 1):
        output_dir = "data/echen333/exit_model"
        model.save_pretrained(save_directory=output_dir)
        tokenizer.save_pretrained(save_directory=output_dir)


        # if step 

        


    # assert train_batch_size % cfg["gradient_accumulation_steps"] == 0, (
    # "train_batch_size must be divisible by gradient_accumulation_steps"
    # )






if __name__ == "__main__":
    main()
