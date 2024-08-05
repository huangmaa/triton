import torch
import triton
import triton.language as tl
import math


# flash attention v1

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, L, M,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen, d,
    BLOCK_SIZE_M:tl.constexpr, # Br
    BLOCK_SIZE_N:tl.constexpr, # Bc
    BLOCK_SIZE_K:tl.constexpr, # d的分块
):
    # 一个program计算 
    # 1、Qi @ Kj = Sij,即 [Br,d] @ [d, Bc] = [Br, Bc]
    # 2、m_ij = rowmax(Sij), Pij = exp(Sij-m_ij), l_ij = rowsum(Pij)
    # 3、Oi = (Pij / l_ij) @ Vj,即 [Br, Bc] @ [Bc, d] = [Br, d]

    row_idx  = tl.program_id(0)
    hb_idx = tl.program_id(1)

    # 计算四维的数据(batch_size, seqle, nheads, head_dim)在内存中的偏移
    idx_batch = hb_idx // nheads # 原数据中的第 idx_batch 个的 batch
    idx_head = hb_idx % nheads # 原数据中的第 idx_batch 个的 batch 的第 idx_head 的 head
    
    offset_m = row_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    offset_d = tl.arange(0, BLOCK_SIZE_K)

    q_ptrs = (
        Q + idx_batch * stride_qb + idx_head * stride_qh + (offset_m[:, None] * stride_qm + offset_d[None, :])
    )
    k_ptrs = (
        K + idx_batch * stride_kb + idx_head * stride_kh + (offset_n[:, None] * stride_kn + offset_d[None, :])
    )
    v_ptrs = (
        V + idx_batch * stride_vb + idx_head * stride_vh + (offset_n[:, None] * stride_vn + offset_d[None, :])
    )

    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    # -- compute qk ----
    for block_k in range(0, d, BLOCK_SIZE_K):
        q = tl.load(q_ptrs + block_k)
        k = tl.load(k_ptrs + block_k)
        v = tl.load(v_ptrs + block_k)
        
        s = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        k = tl.trans(k)
        s += tl.dot(q, k)
        
        # -- compute m_ij, p, l_ij --
        m_ij = tl.max(s, 1) # 当前 block 中，每行的局部最大值
        p = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p, 1) # 当前 block 中，每行的rowsum的结果

        # -- update m_i and l_i --
        m_i_new = tl.maximum(m_i, m_ij) # 更新以获得全局最大值
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij # 更新以获得全局rowsum值

        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
   
    # write back l and m
    l_ptrs = L + hb_idx * seqlen + offset_m
    m_ptrs = M + hb_idx * seqlen + offset_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)

    # initialize pointers to output
    out_ptrs = Out + idx_batch * stride_ob + idx_head * stride_oh + offset_m[:, None] * stride_om + offset_d[None, :]
    tl.store(out_ptrs, acc)

    


@triton.jit
def _bwd_kernel():
    pass



class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, softmax_scale=None):
        """
        q, k, v: (batch_size, seqle(论文中的N), nheads, head_dim(论文中的d))
        """
        batch, seqlen, nheads, d = q.shape
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        # softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        
        o = torch.empty_like(q)
        M = 49152 # 213 server 上的GPU shared mem = 49152 byte
        Bc = M / 4 / d
        Br = min(Bc, 64)
        BLOCK_SIZE_K = max(triton.next_power_of_2(d), 16) 
        if Bc > 128:
            Bc = 128
        
        # 用二维的grid进行并行计算，第一个维度对seqlen进行分块，即对N进行分块；第二个维度对batch * nheads分块
        grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_SIZE_M"]), batch * nheads)
        _fwd_kernel[grid](
            q, k, v, o, l, m,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            o.stride(0), o.stride(2), o.stride(1),
            nheads, seqlen, d,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M=Bc,
            BLOCK_SIZE_N=Br
            )

    @staticmethod
    def backward(ctx, do):
        pass

# attention = _attention.apply

def flash_atten(q, k , v):
    batch, seqlen, nheads, d = q.shape
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    # softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    
    o = torch.empty_like(q)
    seqlen_q_rounded = math.ceil(seqlen / 128) * 128
    l = torch.empty((batch, nheads, seqlen_q_rounded))
    m = torch.empty((batch, nheads, seqlen_q_rounded))
    M = 49152 # 213 server 上的GPU shared mem = 49152 byte
    Bc = M / 4 / d
    Br = min(Bc, 64)
    BLOCK_SIZE_K = max(triton.next_power_of_2(d), 16) 
    if Bc > 128:
        Bc = 128
    
    # 用二维的grid进行并行计算，第一个维度对seqlen进行分块，即对N进行分块；第二个维度对batch * nheads分块
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_SIZE_M"]), batch * nheads)
    _fwd_kernel[grid](
        q, k, v, o, l, m,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        nheads, seqlen, d,
        BLOCK_SIZE_M=Bc,
        BLOCK_SIZE_N=Br,
        BLOCK_SIZE_K=BLOCK_SIZE_K
        )


N, d = 1024, 64
q = torch.rand((1, N, 1, d))
k = torch.rand((1, N, 1, d))
v = torch.rand((1, N, 1, d))
flash_atten(q, k, v)
