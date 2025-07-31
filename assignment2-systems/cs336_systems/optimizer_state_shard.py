import torch
import torch.optim as optim
from typing import Any, Callable
import torch.distributed as dist
from torch import Tensor


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank_loads = [0 for _ in range(dist.get_world_size())]

        self.my_params = []
        self.param_to_rank = {}
        self.params = list(params)
        for param in self.params:
            cur_min = min(enumerate(self.rank_loads), key=lambda x: (x[1], x[0]))
            ind = cur_min[0]

            assert isinstance(param, Tensor), "params must be an iterable of tensors"
            self.rank_loads[ind] += param.numel() * param.element_size()

            self.param_to_rank[param] = ind
            if ind == dist.get_rank():
                self.my_params.append(param)

        assert len(self.my_params) > 0, "Current rank has no params"

        self.optimizer = optimizer_cls(params=self.my_params, **kwargs)
        super().__init__([{"params": x, "from_init": True} for x in self.my_params], defaults={})

    def step(self, closure: Callable | None = None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)
        # after step, need to broadcast my updated params
        handles = []
        for param in self.params:
            handle = dist.broadcast(param.data, src=self.param_to_rank[param], async_op=True)
            handles.append(handle)

        for handle in handles:
            handle.wait()

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=False)
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def _add_param(self, param: Tensor):
        cur_min = min(enumerate(self.rank_loads), key=lambda x: (x[1], x[0]))
        ind = cur_min[0]

        assert isinstance(param, Tensor), "param must be a Tensor"
        self.rank_loads[ind] += param.numel() * param.element_size()

        self.param_to_rank[param] = ind
        if ind == dist.get_rank():
            self.optimizer.add_param_group(param)

    def add_param_group(self, param_group: dict[str, Any]):
        if not param_group.get("from_init", False):
            assert False, "Not currently used right now; for manual param groups"
            self._add_param(param_group["params"])
