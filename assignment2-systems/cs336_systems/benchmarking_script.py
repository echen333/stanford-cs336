"""
Benchmarking script could be a lot cleaner by loading in the configs per model size automatically.

And storing results in Dataframe for easy export to latex.
"""

import torch
import timeit
import argparse
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW

import numpy as np
import torch.cuda.nvtx as nvtx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--data_path", type=str, default="cs336_systems/data/TinyStoriesV2-GPT4-valid.npy")
    parser.add_argument("--mixed_precision", type=bool, default=False)
    parser.add_argument("--only_forward", type=bool, default=False)
    parser.add_argument("--context_len", type=int, default=128)
    return parser.parse_args()


def model_pass(model: BasicsTransformerLM, x, only_forward=True, y=None, optimizer=None):
    if not only_forward:
        model.zero_grad(set_to_none=True)

    nvtx.range_push("forward")
    out = model(x)
    nvtx.range_pop()
    if only_forward:
        torch.cuda.synchronize()
        return

    out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
    y = y.flatten(0, 1)
    loss = cross_entropy(out, y)

    nvtx.range_push("backward")
    loss.backward()
    torch.cuda.synchronize()
    nvtx.range_pop()

    if optimizer is not None:
        optimizer.step()


def memory_profile(args):
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    model = BasicsTransformerLM(
        args.vocab_sz, args.context_len, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(args.device)
    optimizer = AdamW(model.parameters())

    only_forward = True if "only_forward" in args.__dir__() and args.only_forward else False
    print("onlyfoward", only_forward)

    BATCH_SIZE = 4

    with torch.amp.autocast("cuda", torch.bfloat16) if args.mixed_precision else nullcontext():
        arr = np.array(torch.randint(0, 10000, (args.context_len * 20,)))
        x, y = get_batch(arr, BATCH_SIZE, args.context_len, args.device)

        print(f"shape x,y {x.shape}, {y.shape}")

        model_pass(model, x, only_forward, y, optimizer)

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


from typing import Callable


def benchmark_function(fn: Callable, *args, num_warmups=5, num_trials=10):
    for _ in range(num_warmups):
        fn(*args)

    torch.cuda.synchronize()
    timer = timeit.Timer(stmt=lambda: fn(*args))
    arr = np.array(timer.repeat(num_trials, number=1))
    torch.cuda.synchronize()

    ret = arr.mean()
    return ret


from cs336_basics.model import scaled_dot_product_attention
from torch import Tensor
import torch.nn as nn
import pandas as pd
from einops import einsum
import torch.nn.functional as F
import math


class scaled_dot_product_module(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, Q, K, V, mask=None):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def attention_benchmark():
    BATCH_SIZE = 8
    head_emb = [16, 32, 64, 128]
    seq_len = [256, 1024, 4096, 8192, 16384]
    from itertools import product

    records = []
    for h, s in product(head_emb, seq_len):
        Q = torch.randn((BATCH_SIZE, s, h), requires_grad=True).to("cuda")
        K = torch.randn((BATCH_SIZE, s, h), requires_grad=True).to("cuda")
        V = torch.randn((BATCH_SIZE, s, h), requires_grad=True).to("cuda")

        sdpa_module = scaled_dot_product_module()
        torch.compile(sdpa_module)

        res = benchmark_function(scaled_dot_product_attention, Q, K, V)
        res3 = benchmark_function(sdpa_module, Q, K, V)

        out: Tensor = sdpa_module(Q, K, V)
        out2: Tensor = sdpa_module(Q, K, V)
        loss = out.sum()
        loss2 = out2.sum()

        mem_in_use = torch.cuda.memory_allocated() / 1e9

        res2 = benchmark_function(loss.backward, None, True)
        res4 = benchmark_function(loss2.backward, None, True)
        record = {
            "head_dim": h,
            "seq": s,
            "forward_time": res,
            "backward_time": res2,
            "compiled_forward_time": res3,
            "compiled_backward_time": res4,
            # "mem_allocated": mem_in_use,
        }
        print(record)
        records.append(record)

    df = pd.DataFrame(records)
    print(df.to_latex(index_names=False, index=False))


def end_to_end_benchmark(args):
    records = []
    BATCH_SIZE = 4

    for size in ["small", "med", "large", "xlarge", "2.7B"]:
        record = {"size": size}
        for compiled in [True, False]:
            SIZE_CONFIGS = {
                "small": (768, 3072, 12, 12),
                "med": (1024, 4096, 24, 16),
                "large": (1280, 5120, 36, 20),
                "xlarge": (1600, 6400, 48, 25),
                "2.7B": (2560, 10240, 32, 32),
            }

            tmp = SIZE_CONFIGS.get(size)
            if tmp is not None:
                args.d_model, args.d_ff, args.num_layers, args.num_heads = tmp

            print(args, type(args))
            model = BasicsTransformerLM(
                args.vocab_sz,
                args.context_len,
                args.d_model,
                args.num_layers,
                args.num_heads,
                args.d_ff,
                args.rope_theta,
            ).to(args.device)
            if compiled:
                model = torch.compile(model)
            optimizer = AdamW(model.parameters())

            with torch.amp.autocast("cuda", torch.bfloat16) if args.mixed_precision else nullcontext():
                arr = np.array(torch.randint(0, 10000, (args.context_len * 20,)))
                x, y = get_batch(arr, BATCH_SIZE, args.context_len, args.device)

                res = benchmark_function(model, x)
                res2 = benchmark_function(model_pass, model, x, False, y, optimizer)

                record[f"{'compiled_' if compiled else ''}fwd"] = res
                record[f"{'compiled_' if compiled else ''}bwd"] = res2

        print(record)
        records.append(record)
    print(records)
    df = pd.DataFrame(records)
    print(df.to_latex(index=False))


def flash_benchmarking():
    from itertools import product
    seqs = [128, 1024, 8192, 32768]
    emb_dims = [16, 64, 128]
    mixed_precisions = [False, True]
    records = []
    import triton
    from cs336_systems.flash_attention_triton import flash_fwd_kernel, FlashAttentionTriton
    from cs336_basics.model import scaled_dot_product_attention
    from cs336_basics.optimizer import AdamW


    BATCH_SIZE = 1
    for seq, emb_dim, mixed in product(seqs, emb_dims, mixed_precisions): 
        Q = torch.randn(BATCH_SIZE, seq, emb_dim, requires_grad=True).to("cuda")
        K = torch.randn(BATCH_SIZE, seq, emb_dim, requires_grad=True).to("cuda")
        V = torch.randn(BATCH_SIZE, seq, emb_dim, requires_grad=True).to("cuda")

        with torch.amp.autocast("cuda", torch.bfloat16) if mixed else nullcontext():
            trit_fwd = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(Q, K, V))
            out = FlashAttentionTriton.apply(Q, K, V)
            trit_bwd = triton.testing.do_bench(lambda: out.sum().backward(retain_graph=True))
            def both_pass():
                out = FlashAttentionTriton.apply(Q, K, V)
                out.sum().backward()
            trit_both = triton.testing.do_bench(both_pass)

            del out

            pytorch_fwd = triton.testing.do_bench(lambda: scaled_dot_product_attention(Q, K, V))
            out = scaled_dot_product_attention(Q, K, V)
            pytorch_bwd = triton.testing.do_bench(lambda: out.sum().backward(retain_graph=True))
            def both_py_pass():
                out = scaled_dot_product_attention(Q, K, V)
                out.sum().backward()
            pytorch_both = triton.testing.do_bench(both_py_pass)

            record = {
                "seq": seq,
                "emb_dim": emb_dim,
                "mixed": mixed,
                "py_fwd": pytorch_fwd,
                "trit_fwd": trit_fwd,
                "py_bwd": pytorch_bwd,
                "trit_bwd": trit_bwd,
                "py_both": pytorch_both,
                "trit_both": trit_both,
            }
            print(record)
            records.append(record)
        del Q, K, V

    df = pd.DataFrame(records)
    print(df)
    print(df.to_latex(index=False))

if __name__ == "__main__":
    args = get_args()
    args.vocab_sz = 10000
    args.rope_theta = 10000
    args.device = "cuda"

    # attention_benchmark()
    # memory_profile(args)
    # end_to_end_benchmark(args)
    flash_benchmarking()
