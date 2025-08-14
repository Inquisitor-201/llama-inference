import triton
import triton.language as tl
import torch
import os
import math
#from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

torch.manual_seed(0)

# 标准的 Attention
def standard_softmax_attention(Q, K, V, causal=False):
    seq_len, d_model = Q.shape
    scale = math.sqrt(d_model)
    
    # 计算注意力分数
    attn_scores = Q @ K.T / scale
    
    # 应用因果掩码（如果启用）
    if causal:
        # 创建下三角掩码（包含对角线）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        # 将掩码外的位置设为负无穷大
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    # 计算注意力权重
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # 计算注意力输出
    attention_output = attn_weights @ V
    
    return attn_weights, attention_output


@triton.jit #(debug=True)
def flash_attentionv2_kernel(
    Q, K, V, O,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    q_ctx_start, q_len, kv_len,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    causal: tl.constexpr = 1,
):
    start_m = tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_hz = 0
    q_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    q = tl.load(Q  + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd,
                mask=q_offsets[:, None] < q_len, other=0.0)
    
    L_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32) - float("inf")
    
    # Initialize acc_o with zeros
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K  + (start_n + kv_offsets)[:, None] * stride_kn + d_offsets[None, :] * stride_kd, mask=(start_n + kv_offsets)[:, None] < kv_len, other=0.0)
        v = tl.load(V  + (start_n + kv_offsets)[:, None] * stride_vn + d_offsets[None, :] * stride_vd, mask=(start_n + kv_offsets)[:, None] < kv_len, other=0.0)
        
        # Manually transpose k
        k_t = tl.trans(k, (1, 0))
        Sij = tl.dot(q, k_t) / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))

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
    tl.store(O + q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od, o, mask=q_offsets[:, None] < q_len)

def flash_attentionv2(q, k, v, Br, Bc):
    #BLOCK = 128
    
    # Output tensor
    o = torch.empty_like(q)
    # Intermediate tensors
    
    grid = (triton.cdiv(q.shape[0], Br), )
    
    flash_attentionv2_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), 
        v.stride(0), v.stride(1), 
        o.stride(0), o.stride(1),
        0, q.shape[0], k.shape[0],
        BLOCK_M=Br, BLOCK_DMODEL=q.shape[1], BLOCK_N=Bc,
        causal=False,  # Set to 1 for causal attention
    )
    
    return o

# 创建示例数据
N_CTX, D_MODEL = 2048, 128
# SM_M = 101376
Q = torch.randn((N_CTX, D_MODEL), device='cuda', dtype=torch.float32)
K = torch.randn((N_CTX, D_MODEL), device='cuda', dtype=torch.float32)
V = torch.randn((N_CTX, D_MODEL), device='cuda', dtype=torch.float32)

seq_length, q_head_dim = Q.shape[0], Q.shape[1]
k_seq_length, k_head_dim = K.shape[0], K.shape[1]
v_seq_length, v_head_dim = K.shape[0], K.shape[1]
assert q_head_dim == k_head_dim
assert k_seq_length == v_seq_length
Br = 32 #min(int(SM_M / 4 / q_head_dim), q_head_dim)
Bc = 32 #int(SM_M / 4 / q_head_dim)

# 调用 Flash Attention
output = flash_attentionv2(Q, K, V, Br, Bc)

_, expected_attention = standard_softmax_attention(Q, K, V)

print("=========compare=========")
print(output.shape,expected_attention.shape)
print(output)
print(expected_attention)
assert torch.allclose(output, expected_attention, atol=1e-2, rtol=0), "Error in flash attention calculation"
print("Hooray! Flash attention calculation is correct!")