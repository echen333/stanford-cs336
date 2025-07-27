import triton
import triton.language as tl
import torch
from torch import Tensor
import pdb
from einops import rearrange

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
    ):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index21
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),order=(1, 0))
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(K_TILE_SIZE,),
        order=(0,),
    )

    one = tl.full((), 1.0, dtype=tl.float32)
    zero = tl.full((), 0.0, dtype=tl.float32)
    
    offs = tl.arange(0, Q_TILE_SIZE)
    tmp_eye = tl.where(offs[:, None] == offs[None, :], one, zero)
    O_old = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    m_old = tl.zeros((Q_TILE_SIZE,), tl.float32)
    l_old = tl.zeros((Q_TILE_SIZE,), tl.float32)

    q_inds = tl.arange(0, Q_TILE_SIZE) 
    for j in range(Tk):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(j * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),order=(1, 0))

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(j * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),order=(1, 0))

        Q_block = tl.load(Q_block_ptr) # boundary check?
        
        K_block = tl.load(K_block_ptr) # need mask
        V_block = tl.load(V_block_ptr)

        tl.device_assert(len(K_block.shape) == 2)
        K_T = tl.trans(K_block) # why cant i give dimensions?
        S_ij = tl.dot(Q_block, K_T) * scale

        k_inds = tl.arange(0, K_TILE_SIZE)
        if IS_CAUSAL:
            neg_inf = tl.full((), -1e6, dtype=tl.float32)
            S_ij = tl.where(q_inds[:, None] + Q_TILE_SIZE * query_tile_index >= k_inds[None, :] + j * K_TILE_SIZE, S_ij, neg_inf)

        tmp = tl.max(S_ij, axis=-1, return_indices=False)
        m_ij = tl.maximum(m_old, tmp)

        P_ij = tl.exp(S_ij - tl.expand_dims(m_ij, -1))
        l_ij = tl.exp(m_old - m_ij) * l_old + tl.sum(P_ij, -1)

        delta = tl.exp(m_old - m_ij)
        delta_diag = tl.expand_dims(delta, -1) * tmp_eye

        O_ij = tl.dot(delta_diag, O_old) + tl.dot(P_ij, V_block)

        O_old, l_old, m_old = O_ij, l_ij, m_ij
    
    tmp2 = tl.div_rn(one, l_old)
    tmp2_diag = tl.expand_dims(tmp2, -1) * tmp_eye
    Oi = tl.dot(tmp2_diag, O_old)
    Li = m_old + tl.log(l_old)

    tl.store(O_block_ptr, Oi)
    tl.store(L_block_ptr, Li)
        

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq, Bk = 32, 32
        b, Nq, D = Q.shape
        Nk = K.shape[1]

        L = torch.empty(b, Nq).to("cuda")
        O = torch.empty(b, Nq, D).to("cuda")

        print(K.shape, "K shape")
        print(Q.shape, "Q shape")
        flash_fwd_kernel[(triton.cdiv(Nq, Bq), b)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            D ** -0.5,
            D, Bq, Bk, is_causal
            )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        print(O)
        print(torch.sum(O.eq(torch.zeros_like(O))))
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        is_causal = ctx.is_causal

        D = torch.sum(O * dO, -1)
        d = Q.shape[-1]
        K_T = rearrange(K, "b s d -> b d s")
        S = Q @ K_T * (d**-0.5)
        
        if is_causal:
            seq = S.shape[1]
            mask = ~torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).to(device=S.device)
            S = S.masked_fill(mask == 0, -float('inf'))
        P = torch.exp(S - L.unsqueeze(-1))

        dV = rearrange(P, "b s d -> b d s") @ dO
        dP = dO @ rearrange(V, "b s d -> b d s")
        dS = P * (dP - D.unsqueeze(-1))
        dQ = dS @ K * (d**-0.5)
        dK = rearrange(dS, "b s d -> b d s") @ Q * (d**-0.5)

        return dQ, dK, dV, None