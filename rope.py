import torch

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

def _calc_rope_freqs(seq_len, head_dim, device):
    # 计算频率（仅计算前半部分）
    theta = (10000.0 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)).unsqueeze(0)
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    freqs = seq_idx * theta
    return freqs.cos(), freqs.sin()

def _rope(x, _cos, _sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    rotated = torch.cat([x1 * _cos - x2 * _sin, x2 * _cos + x1 * _sin], dim=-1)
    return rotated

if __name__ == '__main__':
    BATCH_SIZE, NH, N_CTX, DI = 4, 4, 256, 32
    x = torch.randn(BATCH_SIZE, NH, N_CTX, DI)
    _cos, _sin = _calc_rope_freqs(N_CTX, DI, torch.device('cpu'))
    y = _rope(x, _cos, _sin)
    y_standard = standard_rope(x)
    print(y)
    print(y_standard)
    assert torch.allclose(y, y_standard, atol=1e-4), "RoPE implementation mismatch"
    print("RoPE applied successfully.")
