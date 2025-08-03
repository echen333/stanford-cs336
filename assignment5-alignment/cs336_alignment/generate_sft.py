from openai import OpenAI
import os
from dotenv import load_dotenv
from datasets import load_dataset
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

MODEL_ENDPOINT = "https://openrouter.ai/api/v1/completions"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
# dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")

arr = []


def ask_model(problem):
    messages = [{"role": "user", "content": problem}]
    completions = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
    )
    prompt = problem
    response_text = completions.choices[0].message.content
    return prompt, response_text


def get_prompt_template(path: str):
    with open(path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def get_prompt(prompt_template: str, question):
    return prompt_template.format(question=question)


prompt_template = get_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
with ThreadPoolExecutor(10) as executor:
    futures = [
        executor.submit(ask_model, get_prompt(prompt_template, item["problem"]))
        for item in dataset.select(range(1024))
    ]

    for future in as_completed(futures):
        (
            problem,
            response,
        ) = future.result()
        print(f"finished {problem}")
        arr.append({"prompt": problem, "response": response})

print("arr", arr)

path = "data/my_sft.jsonl"
with open(path, "w") as f:
    for obj in arr:
        f.write(json.dumps(obj) + "\n")
