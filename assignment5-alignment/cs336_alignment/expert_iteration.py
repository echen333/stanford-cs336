import torch
import wandb
from vllm import SamplingParams, LLM
from datasets import load_dataset
from torch.optim import AdamW
from cs336_alignment.utils import (
    run_sft_microbatch_train_step,
    run_tokenize_prompt_and_output,
    get_response_log_probs,
    load_policy_into_vllm_instance,
    init_vllm,
)
from cs336_alignment.math_baseline import get_prompt, get_prompt_template
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_sampling_worker(output_dir, sampling_params, dataset, exit_batch_size, step):
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    print(f"running eval worker now {output_dir}")
    llm: LLM = init_vllm(output_dir, "cuda:0", 42)
    load_policy_into_vllm_instance(model, llm)

    prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")

    NUM_ITEMS = exit_batch_size
    prompts = [get_prompt(prompt_template, x) for x in dataset["problem"][:NUM_ITEMS]]
    labels = dataset["solution"][:NUM_ITEMS]
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)

    data_arr = []
    for output, label in zip(outputs, labels):
        for ind in range(len(output.outputs)):
            reward = r1_zero_reward_fn(output.outputs[ind].text, label)
            if reward["reward"] > 0.99:
                data_arr.append(
                    {"problem": output.prompt, "solution": output.outputs[ind].text}
                )
                break

    acc = len(data_arr) / NUM_ITEMS
    torch.cuda.empty_cache()

    return acc, data_arr


def run_sft(cfg, model, all_input_ids, all_labels, response_mask):
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    for step in range(1, cfg["sft_steps"] + 1):
        inds = torch.randint(0, len(all_input_ids), (cfg["sft_batch_size"],))
        input_ids = all_input_ids[inds]
        labels = all_labels[inds]
        mask = response_mask[inds]

        print(f"step {step}")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = get_response_log_probs(model, input_ids, labels, True)

            logits = res["log_probs"]
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Step {step}: NaN/inf found in model logits!")

            avg_entropy = torch.mean(res["token_entropy"])

            loss, _ = run_sft_microbatch_train_step(
                logits, mask, cfg["gradient_accumulation_steps"]
            )

        if step % cfg["gradient_accumulation_steps"] == 0:
            print("CLIP AND STEP")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        wandb_log = {
            "sft/loss": loss.item(),
            "sft/entropy": avg_entropy,
            "sft": step,
        }

        wandb.log(wandb_log)


def main():
    cfg = {
        "gradient_accumulation_steps": 16,
        "exit_batch_size": 512,
        "sft_batch_size": 1,
        "sft_steps": 96,
        "device": "cuda",
        "lr": 2e-4,
        "seed": 42,
        "n_ei_steps": 5,
        "G": 5,
    }
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])

    run = wandb.init(entity="eddys", project="stanford-lm-5", config=cfg, group="exit")

    wandb.define_metric("exit_step")
    # wandb.define_metric("eval/*", step_metric="eval_step")

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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

    wandb.define_metric("exit_step")
    wandb.define_metric("sft_step")
    wandb.define_metric("exit/*", step_metric="exit_step")
    wandb.define_metric("sft/*", step_metric="sft_step")

    for exit_step in range(1, cfg["n_ei_steps"] + 1):
        output_dir = "data/echen333/exit_model"
        model.save_pretrained(save_directory=output_dir)
        tokenizer.save_pretrained(save_directory=output_dir)

        model.to("cpu")
        torch.cuda.empty_cache()

        dataset.shuffle(seed=exit_step)
        avg_reward, data_arr = run_sampling_worker(
            output_dir, sampling_params, dataset, cfg["exit_batch_size"], exit_step
        )
        assert len(data_arr) != 0, "found no successful trajectories!"
        print("finished sampling", avg_reward)

        prompt_strs = [x["problem"] for x in data_arr]
        output_strs = [x["solution"] for x in data_arr]

        tokenized = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        print("len", input_ids.shape, labels.shape)
        mask = tokenized["response_mask"].to(device)

        model.to("cuda")
        torch.cuda.empty_cache()
        run_sft(cfg, model, input_ids, labels, mask)

        wandb_log = {"exit/reward": avg_reward, "exit/exit_step": exit_step}
        wandb.log(wandb_log)


if __name__ == "__main__":
    main()
