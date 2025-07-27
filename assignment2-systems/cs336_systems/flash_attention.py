import torch
from torch import Tensor
import pdb
from einops import rearrange


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
        Bq, Bk = 32, 32

        b, _, d = Q.shape
        assert Q.ndim == 3
        Qs = Q.split(Bq, dim=1)
        Ks = K.split(Bk, dim=1)
        Vs = V.split(Bk, dim=1)
        Tq, Tk = len(Qs), len(Ks)

        def my_diagflat(x: Tensor):
            assert x.ndim == 2
            return x.unsqueeze(2).expand(-1, -1, x.size(1)) * torch.eye(x.shape[1])

        O, L = torch.Tensor(), torch.Tensor()
        for i in range(Tq):
            Qi = Qs[i]
            O_old, l_old, m_old = torch.zeros(b, Bq, d), torch.zeros(b, Bq), torch.ones(b, Bq) * float("-inf")

            for j in range(Tk):
                Kj, Vj = Ks[j], Vs[j]
                Kj_T = rearrange(Kj, "b tq d -> b d tq")
                S_ij = Qi @ Kj_T * (d**-0.5)
                m_ij = torch.maximum(m_old, torch.max(S_ij, dim=-1)[0])

                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                l_ij = torch.exp(m_old - m_ij) * l_old + torch.sum(P_ij, -1)

                delta = torch.exp(m_old - m_ij)
                tmp = my_diagflat(delta)
                O_ij = tmp @ O_old + P_ij @ Vj  # ??

                O_old, l_old, m_old = O_ij, l_ij, m_ij

            Oi = my_diagflat(1 / l_old) @ O_old
            Li = m_old + torch.log(l_old)

            O = torch.concat([O, Oi], dim=1)
            L = torch.concat([L, Li], dim=-1)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
