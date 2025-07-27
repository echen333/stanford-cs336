import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np
import time
import pandas as pd


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_all_reduce_bench(rank, world_size, arr_size):
    setup(rank, world_size)
    data = torch.randn(arr_size)

    # torch.cuda.synchronize()
    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    elapsed = time.time() - start_time
    # print(f"rank {rank} has elapsed {elapsed}")
    obj_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj_list, elapsed)
    # torch.cuda.synchronize()

    if rank == 0:
        arr = np.array(obj_list)
        print(f"mean time: {arr.mean()}")


def benchmark(fn, *args, num_warmups=5, num_trials=10):
    for _ in range(num_warmups):
        fn(*args)

    timer = timeit.Timer(stmt=lambda: fn(*args))
    arr = np.array(timer.repeat(num_trials, number=1))
    return arr.mean()


def all_reduce_bench():
    for nums in [int(x) for x in [25e4, 25e5, 25e6, 25e7]]:
        for world_size in [2, 4, 6]:

            def launch(world_size, nums):
                mp.spawn(distributed_all_reduce_bench, args=(world_size, nums), nprocs=world_size, join=True)

            arr_size = 4 * nums / 1e6
            print(f"each object size is {arr_size} MB with world size {world_size}")

            benchmark(launch, world_size, nums)


def naive_ddp():
    pass


if __name__ == "__main__":
    all_reduce_bench()
