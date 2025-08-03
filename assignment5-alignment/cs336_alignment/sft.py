import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from cs336_alignment.utils import run_sft_microbatch_train_step, log_generations, get_response_log_probs, run_tokenize_prompt_and_output, load_policy_into_vllm_instance, init_vllm
import json
from torch.optim import AdamW
import pdb
from cs336_alignment.math_baseline import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from datasets import load_dataset
import multiprocessing as mp

def run_eval_worker(output_dir, step, dataset, sampling_params):
    model = AutoModelForCausalLM.from_pretrained(output_dir).to("cuda:1")
    print(f"running eval worker now {output_dir}")
    llm = init_vllm(output_dir, "cuda:1", 42)
    load_policy_into_vllm_instance(model, llm)

    df, avg_reward = evaluate_vllm(llm, r1_zero_reward_fn, dataset["problem"][:20], dataset["solution"][:20], eval_sampling_params=sampling_params)
    print(f"finished evluating")

    print(df)
    # wandb.log({
    #     "eval_step": step,
    #     "eval/avg_reward": avg_reward,
    #     "eval/format_reward": df["format_reward"].mean()
    # })

def main():
    cfg = {
        "gradient_accumulation_steps": 1, 
        "batch_size": 16,
        "train_steps": 300,
        "device": "cuda:0",
        "lr": 2e-3,
        "seed": 42,
        "validation_steps": 50,
    }
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    output_dir = "data/echen333/model"
    eval_proc = None

    run = wandb.init(entity="eddys", project="stanford-lm-5", config=cfg)
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    optimizer = AdamW(model.parameters(), cfg["lr"])

    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation

    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")

    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    sft_data_path = "data/my_sft.jsonl"
    data = []
    with open(sft_data_path, "r") as f:
        obj = f.readline()
        data.append(json.loads(obj))

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    dataset = dataset.shuffle(seed=42)

    prompt_strs = [x["prompt"] for x in data]
    output_strs = [x["response"] for x in data]
    tokenized = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)


    # ctx = mp.get_context("spawn")
    # p = ctx.Process(target=run_eval_worker, args=(output_dir, 1.0 / cfg["validation_steps"], dataset, sampling_params))
    # p.start()

    # return
    for step in range(1, cfg["train_steps"] + 1):

        res = get_response_log_probs(model, input_ids, labels, True)
        log_probs = res["log_probs"]
        avg_entropy = torch.mean(res["token_entropy"])

        loss, _ = run_sft_microbatch_train_step(log_probs, response_mask, cfg["gradient_accumulation_steps"])

        if step % cfg["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
        # print(f"loss {loss.item()}")

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

            print(f"starting new saving and process")
            model.save_pretrained(save_directory=output_dir)
            tokenizer.save_pretrained(save_directory=output_dir)

            ctx = mp.get_context("spawn")
            p = ctx.Process(target=run_eval_worker, args=(output_dir, step / cfg["validation_steps"], dataset, sampling_params))
            p.start()


        # print(entropy.dtype, log_probs)
        wandb.log(wandb_log)
        
if __name__ == "__main__":
    main()

