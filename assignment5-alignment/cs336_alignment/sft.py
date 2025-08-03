import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from cs336_alignment.utils import (
    run_sft_microbatch_train_step,
    log_generations,
    get_response_log_probs,
    run_tokenize_prompt_and_output,
    load_policy_into_vllm_instance,
    init_vllm,
)
import json
from torch.optim import AdamW
import pdb
from cs336_alignment.math_baseline import evaluate_vllm, get_prompt, get_prompt_template
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from datasets import load_dataset
import multiprocessing as mp


def run_eval_worker(output_dir, step, dataset, sampling_params, run_id):

    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    print(f"running eval worker now {output_dir}")
    llm = init_vllm(output_dir, "cuda:1", 42)
    load_policy_into_vllm_instance(model, llm)

    prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
    
    NUM_ITEMS = 500
    df, avg_reward = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        [get_prompt(prompt_template, x) for x in dataset["problem"][:NUM_ITEMS]],
        dataset["solution"][:NUM_ITEMS],
        eval_sampling_params=sampling_params,
    )
    print(f"finished evluating")

    logging_path = "data/wandb_tmp.jsonl"
    with open(logging_path, "w") as f:
        obj = {
            "eval_step": step,
            "eval/avg_reward": avg_reward,
            "eval/format_reward": df["format_reward"].mean()
        }
        f.write(json.dumps(obj) + '\n')


def main():
    cfg = {
        "gradient_accumulation_steps": 16,
        "batch_size": 1,
        "train_steps": 1800,
        "device": "cuda:0",
        "lr": 2e-4,
        "seed": 42,
        "validation_steps": 120,
        "dataset_size": 128
    }
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    output_dir = "data/echen333/model"
    eval_proc = None

    run = wandb.init(entity="eddys", project="stanford-lm-5", config=cfg, group="sft")
    run_id = run.id

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    torch.cuda.set_device(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    optimizer = AdamW(model.parameters(), cfg["lr"])

    wandb.define_metric("train_step")  
    wandb.define_metric("eval_step")  
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    torch.autograd.set_detect_anomaly(True)

    sft_data_path = "data/my_sft.jsonl"
    data = []
    with open(sft_data_path) as f:
        while True:
            obj = f.readline()
            if len(obj.strip()) == 0:
                break
            data.append(json.loads(obj))
    data = data[:cfg["dataset_size"]]
    print("ELN DATA", len(data))

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    dataset = dataset.shuffle(seed=cfg["seed"])

    prompt_strs = [x["prompt"] for x in data]
    output_strs = [x["response"] for x in data]
    tokenized = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    all_input_ids = tokenized["input_ids"].to(device)
    all_labels = tokenized["labels"].to(device)
    print("len", all_input_ids.shape, all_labels.shape)
    response_mask = tokenized["response_mask"].to(device)

    for step in range(1, cfg["train_steps"] + 1):
        print(f"step {step}")
        inds = torch.randint(0, len(all_input_ids), (cfg["batch_size"], ))
        input_ids = all_input_ids[inds]
        labels = all_labels[inds]
        mask = response_mask[inds]

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
            model.save_pretrained(save_directory=output_dir)
            tokenizer.save_pretrained(save_directory=output_dir)

            ctx = mp.get_context("spawn")
            eval_proc = ctx.Process(
                target=run_eval_worker,
                args=(
                    output_dir,
                    step,
                    dataset,
                    sampling_params,
                    run_id
                ),
            )
            eval_proc.start()


        wandb.log(wandb_log)
    print("going through wandb log for final time now")
    with open("data/wandb_tmp.jsonl", "r") as f:
        objs = f.readlines()
        for obj in objs:
            wandb.log(json.loads(obj))
    



if __name__ == "__main__":
    import gc
    import torch
    gc.collect()
    main()
