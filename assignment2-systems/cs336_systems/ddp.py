import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np
import time
import torch.nn as nn
import pandas as pd
from functools import partial


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
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
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()


def distributed_all_reduce_bench(rank, world_size, arr_size, backend):
    device = _setup_process_group(rank, world_size, backend)
    data = torch.randn(arr_size).to(device)

    torch.cuda.synchronize()
    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    # print(f"rank {rank} has elapsed {elapsed}")
    obj_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj_list, elapsed)

    if rank == 0:
        arr = np.array(obj_list)
        print(f"mean time: {arr.mean()}")
    
    _cleanup_process_group()


def benchmark(fn, *args, num_warmups=5, num_trials=10):
    for _ in range(num_warmups):
        fn(*args)

    timer = timeit.Timer(stmt=lambda: fn(*args))
    arr = np.array(timer.repeat(num_trials, number=1))
    return arr.mean()


def all_reduce_bench():
    tests = [("nccl", 25e4, 2), ("nccl", 25e5, 2), ("nccl", 25e6, 2), ("nccl", 25e7, 2)]
    tests.extend([
            ("gloo", 25e7, 2), ("gloo", 25e7, 4), ("gloo", 25e7, 6),
            ("nccl", 25e7, 2), ("nccl", 25e7, 4), ("nccl", 25e7, 6),
            ])

    for backend, nums, world_size in tests:
        nums = int(nums)
        def launch(world_size, nums):
            mp.spawn(distributed_all_reduce_bench, args=(world_size, nums, backend), nprocs=world_size, join=True)

        arr_size = 4 * nums / 1e6
        print(f"each object size is {arr_size} MB with world size {world_size} with backend {backend}")

        benchmark(launch, world_size, nums)


class DDPOverlapWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for x in module.parameters():
            if x.requires_grad:

                def my_all_reduce(p):
                    p.grad /= dist.get_world_size()
                    handle = dist.all_reduce(p.grad, async_op=True)
                    self.handles.append(handle)

                x.register_post_accumulate_grad_hook(my_all_reduce)

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.handles = []
        cur_bucket_size = 0
        cur_bucket = []
        self.buckets = []
        self.name_to_bucket_ind = {}
        self.total_params = 0

        for ind, (name, x) in enumerate(reversed(list(module.named_parameters()))):
            self.total_params += 1
            if x.requires_grad:
                new_bucket_size = cur_bucket_size + x.numel() * x.element_size() / 1e6

                if new_bucket_size > bucket_size_mb:
                    self.buckets.append(cur_bucket)
                    cur_bucket = []

                cur_bucket.append(ind)
                self.name_to_bucket_ind[name] = len(self.buckets)

                def my_all_reduce(p, ind, bucket_ind):
                    self.bucket_count[bucket_ind] += 1

                    if self.bucket_count[bucket_ind] == len(self.buckets[bucket_ind]):
                        # my job to accumulate since im last in bucket
                        grads = [
                            list(self.module.parameters())[self.total_params - 1 - ind].grad / dist.get_world_size()
                            for ind in self.buckets[bucket_ind]
                        ]
                        flat_grad = _flatten_dense_tensors(grads)
                        handle = dist.all_reduce(flat_grad, async_op=True)
                        unflat = _unflatten_dense_tensors(flat_grad, grads)

                        for param_grad, val in zip(
                            grads,
                            unflat
                        ):
                            param_grad = val

                        self.handles.append(handle)

                x.register_post_accumulate_grad_hook(partial(my_all_reduce, ind=ind, bucket_ind=len(self.buckets)))

        if len(cur_bucket) > 0:
            self.buckets.append(cur_bucket)
        self.bucket_count = [0 for _ in range(len(self.buckets))]

        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        self.bucket_count = [0 for _ in range(len(self.buckets))]

if __name__ == "__main__":
    all_reduce_bench()
