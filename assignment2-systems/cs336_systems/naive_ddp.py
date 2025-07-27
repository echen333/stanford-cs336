import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch.nn as nn
from copy import deepcopy
from torch import Tensor


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = device_count % world_size
        else:
            raise ValueError("Cuda device not found")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


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


def _naive_ddp(rank, world_size, model_cls):
    device = _setup_process_group(rank, world_size, "gloo")
    dist.barrier()

    torch.manual_seed(rank)

    non_parallel_model = model_cls().to(device)
    non_parallel_optim = torch.optim.SGD(non_parallel_model.parameters())

    ddp_model = deepcopy(non_parallel_model)
    ddp_optim = torch.optim.SGD(ddp_model.parameters())

    # broadcast to other from rank 0
    for param in ddp_model.parameters():
        dist.broadcast(param, src=0)
    for param in non_parallel_model.parameters():
        dist.broadcast(param, src=0)

    # Optimizing on same data
    all_x = torch.load("cs336_systems/data/all_x_fixture.pt")
    all_y = torch.load("cs336_systems/data/all_y_fixture.pt")

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

        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        grad = _flatten_dense_tensors([x.grad for x in ddp_model.parameters()])
        dist.all_reduce(grad, async_op=False)
        for param, val in zip(
            ddp_model.parameters(), _unflatten_dense_tensors(grad, [x.grad for x in ddp_model.parameters()])
        ):
            param.grad = val
            param.grad /= world_size

        ddp_optim.step()

        if rank == 0:
            for (non_parallel_param), (ddp_param) in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                assert torch.allclose(non_parallel_param, ddp_param)

        # have to give it the same seed after
        torch.manual_seed(42 + i)
        shuffled_indices = torch.randperm(all_x.size(0))
        all_x = all_x[shuffled_indices]
        all_y = all_y[shuffled_indices]


def naive_ddp():
    torch.random.manual_seed(42)
    world_size = 2

    mp.spawn(_naive_ddp, args=(world_size, ToyModel), nprocs=2, join=True)


if __name__ == "__main__":
    naive_ddp()
