import torch
import wandb
import multiprocessing as mp
from vllm import SamplingParams
from datasets import load_dataset
import json
from torch.optim import AdamW
from cs336_alignment.utils import run_sft_microbatch_train_step, run_tokenize_prompt_and_output, get_response_log_probs, load_policy_into_vllm_instance, init_vllm
from cs336_alignment.math_baseline import get_prompt, get_prompt_template, evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb

def run_sampling_worker(output_dir, sampling_params, dataset, exit_batch_size, step):
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    print(f"running eval worker now {output_dir}")
    llm = init_vllm(output_dir, "cuda:1", 42)
    load_policy_into_vllm_instance(model, llm)

    prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
    
    NUM_ITEMS = exit_batch_size
    df, avg_reward = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        [get_prompt(prompt_template, x) for x in dataset["problem"][:NUM_ITEMS]],
        dataset["solution"][:NUM_ITEMS],
        eval_sampling_params=sampling_params,
    )
    return df, avg_reward


def run_sft(cfg, model, all_input_ids, all_labels, response_mask):
    optimizer = AdamW(model.parameters())
    for step in range(1, cfg["sft_steps"] + 1):
        inds = torch.randint(0, len(all_input_ids), (cfg["sft_batch_size"], ))
        input_ids = all_input_ids[inds]
        labels = all_labels[inds]
        mask = response_mask[inds]

        print(f"step {step}")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = get_response_log_probs(model, input_ids, labels, True)


            logits = res["log_probs"]
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Step {step}: NaN/inf found in model logits!")

                breakpoint()
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
            "train/loss": loss.item(),
            "train/entropy": avg_entropy,
            "train_step": step,
        }

        if step % cfg["validation_steps"] == 0:
            if eval_proc is not None:
                print(f"Waiting for last eval process to finish.")
                eval_proc.join()
                eval_proc = None

            if step != cfg["validation_steps"]:
                print("going through wandb log now")
                with open("data/wandb_tmp.jsonl", "r") as f:
                    objs = f.readlines()
                    for obj in objs:
                        wandb.log(json.loads(obj))

            print(f"starting new saving and process")

        wandb.log(wandb_log)

def main():
    cfg = {
        "sft_gradient_accumulation_steps": 16,
        "exit_batch_size": 2048,
        "sft_batch_size": 1,
        "sft_steps": 180,
        "device": "cuda",
        "lr": 2e-4,
        "seed": 42,
        "dataset_size": 1024,
        "n_ei_steps": 5,
        "G": 5
    }
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])

    run = wandb.init(entity="eddys", project="stanford-lm-5", config=cfg, group="exit")

    wandb.define_metric("exit_step")  
    # wandb.define_metric("eval/*", step_metric="eval_step")

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_min_tokens = 4
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"], n=cfg["G"], seed=cfg["seed"], min_tokens=sampling_min_tokens
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    dataset = dataset.shuffle(seed=cfg["seed"])

    for exit_step in range(1, cfg["n_ei_steps"] + 1):

        dataset.shuffle(seed=exit_step)
        df, avg_reward = run_sampling_worker(None,sampling_params, dataset, cfg["exit_batch_size"], exit_step)
        print("finished sampling", df.head(), avg_reward)

        breakpoint()
        inds = df[df["reward"] > 0.99].index
        prompt_strs = [dataset["prompts"][ind] for ind in inds]
        output_strs = [dataset["solution"][ind] for ind in inds]

        tokenized = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        print("len", input_ids.shape, labels.shape)
        mask = tokenized["response_mask"].to(device)

        run_sft(cfg, model, input_ids, labels, mask)

        wandb_log = {
            "reward": avg_reward,
        }
        wandb.log(wandb_log)


        


if __name__ == "__main__":
    main()
