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
    os.environ["MASTER_PORT"] = "29510"
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
        d_model= 2560,
        d_ff= 10240,
        num_layers= 32,
        num_heads= 32,
        context_len= 128,
        vocab_sz= 10000,
        rope_theta=10000,
        batch_size=64,
    )
    return args

def _train_xl_ddp_benchmark(rank, world_size, model_cls, args):
    device = _setup_process_group(rank, world_size, "nccl")
    dist.barrier()
    torch.manual_seed(rank)

    model = model_cls(
        args.vocab_sz, args.context_len, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(device)
    optimizer = AdamW(model.parameters())

    arr = np.array(torch.randint(0, 10000, (args.context_len * 20,)))
    x, y = get_batch(arr, args.batch_size, args.context_len, device)
    local_bs = x.size(0) // world_size
    offset = rank * local_bs

    for i in range(5):
        arr = np.array(torch.randint(0, 10000, (args.context_len * 20,)))

        print(f"shape x,y {x.shape}, {y.shape}")

        ddp_data = x[offset : offset + local_bs, :]
        ddp_labels = y[offset : offset + local_bs, :]
        out = model(ddp_data)
        out = out.flatten(0,1)
        ddp_labels = ddp_labels.flatten(0,1)
        ddp_loss = cross_entropy(out, ddp_labels)
        ddp_loss.backward()

        torch.cuda.synchronize()
        start_time = time.time()
        for param in model.parameters():
            dist.all_reduce(param.grad, async_op=False)
            param.grad /= world_size
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"elapsed communicating {elapsed}")

        optimizer.step()
        torch.cuda.synchronize()

        torch.manual_seed(42 + i)
        shuffled_indices = torch.randperm(x.size(0))
        x = y[shuffled_indices]
        x = y[shuffled_indices]

        for p in model.parameters():
            print(f"ind {i} has params {p.view(-1)[:5]}")
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

        # from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        # grad = _flatten_dense_tensors([x.grad for x in ddp_model.parameters()])
        # dist.all_reduce(grad, async_op=False)
        # for param, val in zip(
        #     ddp_model.parameters(), _unflatten_dense_tensors(grad, [x.grad for x in ddp_model.parameters()])
        # ):
        #     param.grad = val
        #     param.grad /= world_size

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


def naive_ddp():
    torch.random.manual_seed(42)
    world_size = 2

    # mp.spawn(_naive_ddp, args=(world_size, ToyModel), nprocs=2, join=True)
    args = get_args()
    mp.spawn(_train_xl_ddp_benchmark, args=(world_size, BasicsTransformerLM, args), nprocs=2, join=True)


if __name__ == "__main__":
    naive_ddp()
