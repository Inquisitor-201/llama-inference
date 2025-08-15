import triton
import triton.language as tl
import torch
import os
import math
#from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

torch.manual_seed(0)

def standard_rope(x, theta=10000.0):
    # x: [batch_size, num_heads, seq_len, head_dim]
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # 生成频率（仅计算前半部分）
    theta = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)[:head_dim//2] / head_dim))
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    freqs = seq_idx * theta  # [seq_len, dim//2]

    # 计算cos和sin（前后半部分共享）
    cos = freqs.cos()  # [seq_len, dim//2]
    sin = freqs.sin()

    # 拆分前后半部分
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]

    # 旋转并合并
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated

def standard_softmax_attention(q, k, v, use_rope=False, causal=False):
    batch_size, q_heads, seq_len, head_dim = q.shape
    k_heads = k.shape[1]

    if use_rope:
        q = standard_rope(q)
        k = standard_rope(k)

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

def _calc_rope_freqs(seq_len, head_dim, theta=10000.0, device='cuda'):
    # 计算频率（仅计算前半部分）
    _theta = (theta ** (-torch.arange(0, head_dim, 2, device=device) / head_dim))
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    freqs = seq_idx * _theta
    return freqs.cos(), freqs.sin()

def _rope(x, _cos, _sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    rotated = torch.cat([x1 * _cos - x2 * _sin, x2 * _cos + x1 * _sin], dim=-1)
    return rotated

# @triton.jit
# def _rope(x, cos, sin, BLOCK_DMODEL: tl.constexpr):
#     """
#     RoPE (Rotary Position Embedding) 实现
#     参数:
#         x: 输入张量 [BLOCK_M, BLOCK_DMODEL]
#         cos: cos 频率参数 [BLOCK_M, BLOCK_DMODEL//2]
#         sin: sin 频率参数 [BLOCK_M, BLOCK_DMODEL//2]
#         BLOCK_DMODEL: 头维度大小（常量）
#     """
#     # 计算前半部分和后半部分的偏移量
#     # x1_offsets = tl.arange(0, BLOCK_DMODEL // 2)
#     # x2_offsets = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)
    
#     # 提取输入张量的前后两部分
#     x1 = x[:, :BLOCK_DMODEL // 2]  # [BLOCK_M, half_d]
#     x2 = x[:, BLOCK_DMODEL // 2:]  # [BLOCK_M, half_d]
    
#     # 应用旋转位置编码公式:
#     # rotated = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
#     part1 = x1 * cos - x2 * sin
#     part2 = x2 * cos + x1 * sin
    
#     # 合并两部分结果
#     rotated = tl.cat([part1, part2], axis=1)
#     return rotated

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

    # sc_d_offsets = tl.arange(0, BLOCK_DMODEL // 2)
    # # 加载RoPE的频率参数（COS/SIN）
    # _cos = tl.load(COS + q_offsets[:, None] * BLOCK_DMODEL // 2 + sc_d_offsets[None, :],
    #                 mask=q_offsets[:, None] < q_len, other=1.0)
    # _sin = tl.load(SIN + q_offsets[:, None] * BLOCK_DMODEL // 2 + sc_d_offsets[None, :],
    #                 mask=q_offsets[:, None] < q_len, other=0.0)

    # q = _rope(q, _cos, _sin, BLOCK_DMODEL)
    # k = _rope(k, _cos, _sin, BLOCK_DMODEL)

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

def flash_attentionv2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor,
                      Br: int, Bc: int,
                      use_rope: bool) -> torch.Tensor:
    batch, q_heads, q_seq_length, q_head_dim = q.shape
    k_heads = k.shape[1]
    q_per_kv = q_heads // k_heads

    if use_rope:
        assert cos.shape[0] == q_seq_length
        assert sin.shape[0] == q_seq_length
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)

    # Output tensor
    o = torch.empty_like(q)
    # Intermediate tensors
    
    grid = (batch, q_heads, triton.cdiv(q_seq_length, Br))
    
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
BATCH_SIZE, N_CTX = 64, 1024
NHQ, NHKV, DI = 8, 4, 16
# SM_M = 101376
Q = torch.randn((BATCH_SIZE, NHQ, N_CTX, DI), device='cuda', dtype=torch.float32)
K = torch.randn((BATCH_SIZE, NHKV, N_CTX, DI), device='cuda', dtype=torch.float32)
V = torch.randn((BATCH_SIZE, NHKV, N_CTX, DI), device='cuda', dtype=torch.float32)

q_batch_size, q_heads, q_seq_length, q_head_dim = Q.shape
k_batch_size, k_heads, k_seq_length, k_head_dim = K.shape
v_batch_size, v_heads, v_seq_length, v_head_dim = V.shape

assert q_batch_size == k_batch_size and k_batch_size == v_batch_size
assert q_heads % k_heads == 0 and k_heads == v_heads
assert q_head_dim == k_head_dim and k_head_dim == v_head_dim
Br = min(64, Q.shape[-2])  # 至少考虑序列长度
Bc = min(64, K.shape[-2])

_cos, _sin = _calc_rope_freqs(q_seq_length, q_head_dim, device='cuda')

# 调用 Flash Attention
output = flash_attentionv2(Q, K, V, _cos, _sin, Br, Bc, use_rope=True)

_, expected_attention = standard_softmax_attention(Q, K, V, use_rope=True)

print("=========compare=========")
print(output.shape,expected_attention.shape)
print(output)
print(expected_attention)

print("Max difference:", torch.max(torch.abs(output - expected_attention)))
print("Mean difference:", torch.mean(torch.abs(output - expected_attention)))
assert torch.allclose(output, expected_attention, atol=1e-2), "Error in flash attention calculation"
print("Hooray! Flash attention calculation is correct!")

# --------------------------------------------

from torch.profiler import profile, record_function, ProfilerActivity

N = 100

for _ in range(N//5):
    # 调用 Flash Attention
    # output = flash_attentionv2(Q, K, V, _cos, _sin, Br, Bc, use_rope=True)
    _, expected_attention = standard_softmax_attention(Q, K, V, use_rope=True)

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA if torch.cuda.is_available() else ProfilerActivity.CPU
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.no_grad():
        _cos, _sin = _calc_rope_freqs(q_seq_length, q_head_dim, device='cuda')

        for _ in range(N):
            # 调用 Flash Attention
            output = flash_attentionv2(Q, K, V, _cos, _sin, Br, Bc, use_rope=True)
            # _, expected_attention = standard_softmax_attention(Q, K, V, use_rope=True)
            
print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))
prof.export_chrome_trace("trace-opt.json")
