import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch.nn as nn
from copy import deepcopy
from torch import Tensor
import time
import numpy as np
import argparse
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29511"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = rank % world_size
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Cuda device not found")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    dist.barrier()
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 30)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(30, 5)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.layer1(x))
        out2 = self.relu2(self.layer2(out1))
        return out2


def get_args():
    from argparse import Namespace

    args = Namespace(
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
        context_len=16,
        vocab_sz=10000,
        rope_theta=10000,
        batch_size=64,
    )
    return args


def _train_xl_ddp_benchmark(rank, world_size, model_cls, args, model_wrap=True, bucket=False):
    device = _setup_process_group(rank, world_size, "nccl")
    dist.barrier()
    torch.manual_seed(rank)

    model = model_cls(
        args.vocab_sz, args.context_len, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(device)
    print(f"initialized {model}")

    from cs336_systems.ddp import DDPOverlapWrapper, DDPOverlapBucketed
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    if model_wrap:
        if bucket:
            model = DDPOverlapBucketed(model, 250)
        else:
            model = DDPOverlapWrapper(model)
    else:
        for param in model.parameters():
            dist.broadcast(param, src=0)

    optimizer = AdamW(model.parameters())

    arr = np.array(torch.randint(0, 10000, (args.context_len * 2,)))
    x, y = get_batch(arr, args.batch_size, args.context_len, device)
    local_bs = x.size(0) // world_size
    offset = rank * local_bs

    sum_elapsed = 0
    num_steps = 20
    print(f"starting model for {num_steps} steps")
    torch.cuda.synchronize()
    model_start_time = time.time()
    for i in range(num_steps):
        ddp_data = x[offset : offset + local_bs, :]
        ddp_labels = y[offset : offset + local_bs, :]
        out = model(ddp_data)
        out = out.flatten(0, 1)
        ddp_labels = ddp_labels.flatten(0, 1)
        ddp_loss = cross_entropy(out, ddp_labels)
        ddp_loss.backward()

        if model_wrap:
            model.finish_gradient_synchronization()
        else:
            torch.cuda.synchronize()
            start_time = time.time()
            for param in model.parameters():
                dist.all_reduce(param.grad, async_op=False)
                param.grad /= world_size

            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_flat = _flatten_dense_tensors(grads)
            dist.all_reduce(grad_flat, async_op=False)
            unflat = _unflatten_dense_tensors(grad_flat, grads)
            for param, val in zip(model.parameters(), unflat):
                # am i actually assigning here?
                param.grad = val
                param.grad /= world_size

            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            sum_elapsed += elapsed

        optimizer.step()

        # print(f"stepp {i}")
        torch.manual_seed(42 + i)
        shuffled_indices = torch.randperm(x.size(0))
        x = x[shuffled_indices]
        y = y[shuffled_indices]

    torch.cuda.synchronize()
    model_elapsed = time.time() - model_start_time
    print(f"avg elapsed time {sum_elapsed / num_steps}")
    print(f"avg model epoch time {model_elapsed / num_steps}")
    _cleanup_process_group()


def _naive_ddp(rank, world_size, model_cls):
    device = _setup_process_group(rank, world_size, "nccl")
    dist.barrier()

    torch.manual_seed(rank)

    non_parallel_model = model_cls().to(device)
    non_parallel_optim = torch.optim.SGD(non_parallel_model.parameters())

    ddp_model = deepcopy(non_parallel_model).to(device)
    ddp_optim = torch.optim.SGD(ddp_model.parameters())

    # broadcast to other from rank 0
    for param in ddp_model.parameters():
        dist.broadcast(param, src=0)
    for param in non_parallel_model.parameters():
        dist.broadcast(param, src=0)

    # Optimizing on same data
    all_x = torch.load("cs336_systems/data/all_x_fixture.pt").to(device)
    all_y = torch.load("cs336_systems/data/all_y_fixture.pt").to(device)

    local_bs = all_x.size(0) // world_size
    offset = rank * local_bs

    loss_fn = nn.MSELoss()

    for param in ddp_model.parameters():
        # print(param)
        print(param.grad)
        break
    for i in range(5):
        non_parallel_optim.zero_grad()
        non_parallel_out = non_parallel_model(all_x)
        non_parallel_loss = loss_fn(non_parallel_out, all_y)
        non_parallel_loss.backward()
        non_parallel_optim.step()

        ddp_optim.zero_grad()
        ddp_data = all_x[offset : offset + local_bs, :]
        ddp_labels = all_y[offset : offset + local_bs, :]
        ddp_out = ddp_model(ddp_data)
        ddp_loss: Tensor = loss_fn(ddp_out, ddp_labels)
        ddp_loss.backward()

        torch.cuda.synchronize()
        start_time = time.time()
        for param in ddp_model.parameters():
            dist.all_reduce(param.grad, async_op=False)
            param.grad /= world_size
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"elapsed communicating {elapsed}")

        ddp_optim.step()

        if rank == 0:
            for (non_parallel_param), (ddp_param) in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                assert torch.allclose(non_parallel_param, ddp_param)

        # have to give it the same seed after
        torch.manual_seed(42 + i)
        shuffled_indices = torch.randperm(all_x.size(0))
        all_x = all_x[shuffled_indices]
        all_y = all_y[shuffled_indices]

    _cleanup_process_group()


def _mp_memory_profile_shard(rank, world_size, args, num_epochs=100, optimizer_shard=False):
    _setup_process_group(rank, world_size, "nccl")

    torch.manual_seed(42)
    torch.cuda.memory._record_memory_history(max_entries=10000000, context=None, stacks="python")
    args.device = "cuda"
    print(f"loading model")
    model = BasicsTransformerLM(
        args.vocab_sz, args.context_len, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(args.device)
    print("loaded model")

    if optimizer_shard:
        from cs336_systems.optimizer_state_shard import OptimizerStateSharding

        optimizer = OptimizerStateSharding(model.parameters(), AdamW)
    else:
        optimizer = AdamW(model.parameters())

    BATCH_SIZE = 4

    print(f"running memory profile for {num_epochs} epochs and shard: {optimizer_shard}")

    arr = np.array(torch.randint(0, 10000, (args.context_len * 20,)))
    all_x, all_y = get_batch(arr, BATCH_SIZE, args.context_len, args.device)

    local_bs = all_x.size(0) // world_size
    offset = rank * local_bs

    model_start_time = time.time()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        ddp_data = all_x[offset : offset + local_bs, :]
        ddp_labels = all_y[offset : offset + local_bs, :]
        ddp_out = model(ddp_data)

        out = ddp_out.flatten(0, 1)
        ddp_labels = ddp_labels.flatten(0, 1)

        ddp_loss = cross_entropy(out, ddp_labels)

        ddp_loss.backward()
        optimizer.step()

    model_elapsed = time.time() - model_start_time
    print(f"avg model epoch time {model_elapsed / num_epochs}")
    print(f"snapshooting now")
    # snap = torch.cuda.memory._snapshot()
    # print(f"[Rank {rank}] memory records: {snap.num_records}")

    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{dist.get_rank()}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f" done done ")
    _cleanup_process_group()


def naive_ddp():
    torch.random.manual_seed(42)
    world_size = 2

    # mp.spawn(_naive_ddp, args=(world_size, ToyModel), nprocs=2, join=True)
    args = get_args()
    # mp.spawn(_train_xl_ddp_benchmark, args=(world_size, BasicsTransformerLM, args, True), nprocs=2, join=True)
    mp.spawn(_mp_memory_profile_shard, args=(world_size, args), nprocs=2, join=True)


if __name__ == "__main__":
    naive_ddp()
