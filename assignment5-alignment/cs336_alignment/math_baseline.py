import os
from vllm import LLM, SamplingParams
from typing import Callable, List
import pickle
import pandas as pd
import datetime
from datasets import load_dataset


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    labels: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    assert len(outputs) == len(labels), "prompts len and labels len must match"

    rewards = []
    for output, label in zip(outputs, labels):
        reward = reward_fn(output.outputs[0].text, label)
        rewards.append(reward)
        print("NEW ITEM", output, "\n LABEL:", label, "\n REWARD:", reward)

    df = pd.DataFrame(rewards)
    print(df)
    os.makedirs("data/", exist_ok=True)
    df.to_pickle(f"data/evaluation_df_{datetime.datetime.now().strftime('%H:%M:%S')}")


def get_prompt_template(path: str):
    with open(path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def get_prompt(prompt_template: str, question):
    return prompt_template.format(question=question)


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)  # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")

    NUM_ITEMS = 50

    # prompts = dataset
    problems = dataset["problem"][:NUM_ITEMS]
    solutions = dataset["solution"][:NUM_ITEMS]

    prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
    prompts = [get_prompt(prompt_template, problem) for problem in problems]
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, solutions, sampling_params)


if __name__ == "__main__":
    main()
