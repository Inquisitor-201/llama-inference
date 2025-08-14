import triton
import triton.language as tl
import torch
import os
import math
#from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

torch.manual_seed(0)

def standard_softmax_attention(q, k, v, causal=False):
    batch_size, q_heads, seq_len, head_dim = q.shape
    k_heads = k.shape[1]

    q_per_kv = q_heads // k_heads

    if q_per_kv > 1:
        k = k.repeat_interleave(q_per_kv, dim=1)
        v = v.repeat_interleave(q_per_kv, dim=1)
        # k/v: [batch_size, num_heads_q, seq_len, head_dim]

    # 缩放点积注意力 + 因果掩码
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    if causal:
        # attn_weights: [batch_size, num_heads_q, seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        # causal_mask: [seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    # attn_output: [batch_size, num_heads_q, seq_len, head_dim]

    return attn_weights, attn_output

@triton.jit #(debug=True)
def flash_attentionv2_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    q_per_kv: tl.constexpr,
    q_ctx_start, q_len, kv_len,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    causal: tl.constexpr = False,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    start_m = tl.program_id(2)
    kvh = qh // q_per_kv

    q_off_hz = b * stride_qb + qh * stride_qh
    k_off_hz = b * stride_kb + kvh * stride_kh
    v_off_hz = b * stride_vb + kvh * stride_vh
    o_off_hz = b * stride_ob + qh * stride_oh

    q_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    q = tl.load(Q + q_off_hz + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd,
                mask=q_offsets[:, None] < q_len, other=0.0)
    
    L_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32) - float("inf")
    
    # Initialize acc_o with zeros
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K + k_off_hz + d_offsets[:, None] * stride_kd + (start_n + kv_offsets)[None, :] * stride_kn, mask=(start_n + kv_offsets)[None, :] < kv_len, other=0.0)
        v = tl.load(V + v_off_hz + (start_n + kv_offsets)[:, None] * stride_vn + d_offsets[None, :] * stride_vd, mask=(start_n + kv_offsets)[:, None] < kv_len, other=0.0)
        
        # Manually transpose k
        # k_t = tl.trans(k, (1, 0))
        Sij = tl.dot(q, k) / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))

        #import pdb; pdb.set_trace()
        
        if causal:
            Sij = tl.where(q_offsets[:, None] + q_ctx_start >= (start_n + kv_offsets)[None, :], Sij, float("-inf"))
        
        mij_hat = tl.max(Sij, 1)[:, None]
        
        Mi_new = tl.where(m_i > mij_hat, m_i, mij_hat)

        Pij =  tl.math.exp(Sij - Mi_new)
        
        Li_new = tl.math.exp(m_i - Mi_new) * L_i + tl.sum(Pij, axis=-1)[:, None]
        
        acc_o = acc_o * tl.math.exp(m_i - Mi_new) + tl.dot(Pij, v)
        
        L_i = Li_new
        m_i = Mi_new
        
        # tl.store(L + off_hz * N_CTX + q_offsets, L_i, mask=q_offsets < N_CTX)
        # tl.store(M + off_hz * N_CTX + q_offsets, m_i, mask=q_offsets < N_CTX)
        
    o = acc_o / L_i
    tl.store(O + o_off_hz + q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od, o, mask=q_offsets[:, None] < q_len)

def flash_attentionv2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Br: int, Bc: int) -> torch.Tensor:
    batch, q_heads, _, _ = q.shape
    k_heads = k.shape[1]
    q_per_kv = q_heads // k_heads

    # Output tensor
    o = torch.empty_like(q)
    # Intermediate tensors
    
    grid = (batch, q_heads, triton.cdiv(q.shape[-2], Br))
    
    flash_attentionv2_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_per_kv,
        0, q.shape[-2], k.shape[-2],
        BLOCK_M=Br, BLOCK_DMODEL=q.shape[-1], BLOCK_N=Bc,
        causal=False,  # Set to 1 for causal attention
    )
    
    return o

# 创建示例数据
BATCH_SIZE, N_CTX = 32, 512
NHQ, NHKV, DI = 8, 4, 16
# SM_M = 101376
Q = 0.1 * torch.randn((BATCH_SIZE, NHQ, N_CTX, DI), device='cuda', dtype=torch.float32)
K = 0.1 * torch.randn((BATCH_SIZE, NHKV, N_CTX, DI), device='cuda', dtype=torch.float32)
V = 10 * torch.randn((BATCH_SIZE, NHKV, N_CTX, DI), device='cuda', dtype=torch.float32)

q_batch_size, q_heads, q_seq_length, q_head_dim = Q.shape
k_batch_size, k_heads, k_seq_length, k_head_dim = K.shape
v_batch_size, v_heads, v_seq_length, v_head_dim = V.shape

assert q_batch_size == k_batch_size and k_batch_size == v_batch_size
assert q_heads % k_heads == 0 and k_heads == v_heads
assert q_head_dim == k_head_dim and k_head_dim == v_head_dim
Br = min(64, Q.shape[-2])  # 至少考虑序列长度
Bc = min(64, K.shape[-2])

# 调用 Flash Attention
output = flash_attentionv2(Q, K, V, Br, Bc)

_, expected_attention = standard_softmax_attention(Q, K, V)

print("=========compare=========")
print(output.shape,expected_attention.shape)
print(output)
print(expected_attention)

print("Max difference:", torch.max(torch.abs(output - expected_attention)))
print("Mean difference:", torch.mean(torch.abs(output - expected_attention)))
assert torch.allclose(output, expected_attention, rtol=1e-2), "Error in flash attention calculation"
print("Hooray! Flash attention calculation is correct!")