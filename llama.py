import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Dict, Any, Tuple
import math
from transformers import AutoTokenizer

# 修复的RMSNorm实现
class FixedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ACTFN_MAP = {
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, theta):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads_q
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.theta = theta

        # 确保query头数能被key/value头数整除，这是GQA的要求
        assert num_heads_q % num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv"
        self.q_per_kv = num_heads_q // num_heads_kv  # 每个kv头对应的q头数量

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads_kv * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads_kv * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def _rope(self, x):
        # x: [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = x.shape
        device = x.device
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # 生成频率（仅计算前半部分）
        theta = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2, device=device)[:head_dim//2] / head_dim))
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

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape

        # 线性投影到多头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads_kv, self.head_dim).transpose(1, 2)

        # q: [batch_size, num_heads_q, seq_len, head_dim]
        # k/v: [batch_size, num_heads_kv, seq_len, head_dim]

        q, k = self._rope(q), self._rope(k)

        # 将 kv 头拓展以匹配 q 头（GQA）
        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)
            # k/v: [batch_size, num_heads_q, seq_len, head_dim]

        # 缩放点积注意力 + 因果掩码
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_weights: [batch_size, num_heads_q, seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        # causal_mask: [seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch_size, num_heads_q, seq_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # attn_output (merged heads): [batch_size, seq_len, hidden_size]
        return self.o_proj(attn_output)

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = ACTFN_MAP[config["hidden_act"]]()
        self.gate_proj = torch.nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = torch.nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = torch.nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class CustomLlamaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]

        # 嵌入层: 将 token id 映射到向量
        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        # embed_tokens.weight: [vocab_size, hidden_size]

        # 自定义实现的Transformer层
        self.layers = torch.nn.ModuleList([
            self.create_decoder_layer(config) for _ in range(config["num_hidden_layers"])
        ])

        # 输出层归一化
        self.norm = FixedRMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # 独立的lm_head层
        self.lm_head = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def create_decoder_layer(self, config):
        """创建包含自定义算子的Transformer层"""
        return torch.nn.ModuleDict({
            "self_attn": GroupedQueryAttention(
                hidden_size=config["hidden_size"],
                num_heads_q=config["num_attention_heads"],
                num_heads_kv=config["num_key_value_heads"],
                theta=config["rope_theta"]
            ),
            "mlp": MLP(config),
            "input_layernorm": FixedRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "post_attention_layernorm": FixedRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        })

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embed_tokens(input_ids)
        # x (embeddings): [batch_size, seq_len, hidden_size]

        for layer in self.layers:
            residual = x
            x = layer["input_layernorm"](x)
            x = layer["self_attn"](x)
            x = residual + x

            residual = x
            x = layer["post_attention_layernorm"](x)
            x = layer["mlp"](x)
            x = residual + x

        x = self.norm(x)
        # x (final hidden): [batch_size, seq_len, hidden_size]
        last_hidden = x[:, -1, :]
        # last_hidden: [batch_size, hidden_size]
        
        # 使用独立的lm_head层
        logits = self.lm_head(last_hidden)
        # logits: [batch_size, vocab_size]
        return logits

# 3. 加载配置和权重的工具函数
def load_llama_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载模型配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_llama_weights(weights_path: str = "model.safetensors") -> Dict[str, torch.Tensor]:
    """加载模型权重文件"""
    return load_file(weights_path, device="cpu")  # 先加载到CPU

def map_weights_to_custom_model(
    custom_model: CustomLlamaModel, 
    original_weights: Dict[str, torch.Tensor]
) -> CustomLlamaModel:
    """将原始权重映射到自定义模型"""

    print("original_weights:", original_weights.keys())

    # 加载权重到自定义模型
    custom_state_dict = custom_model.state_dict()
    mapped_count = 0
    total_count = len(custom_state_dict)
    
    for custom_name in custom_state_dict.keys():
        if custom_name == "lm_head.weight":
            # lm_head权重应该与embed_tokens权重相同（权重绑定）
            if "model.embed_tokens.weight" in original_weights:
                custom_state_dict[custom_name].copy_(original_weights["model.embed_tokens.weight"])
                print(f"✅ 加载权重: model.embed_tokens.weight -> {custom_name}, shape: {original_weights['model.embed_tokens.weight'].shape}")
                mapped_count += 1
            else:
                print(f"❌ 警告: 未找到embed_tokens权重用于lm_head")
        else:
            orig_name = "model." + custom_name
            if orig_name in original_weights:
                # 确保权重形状匹配
                if original_weights[orig_name].shape == custom_state_dict[custom_name].shape:
                    custom_state_dict[custom_name].copy_(original_weights[orig_name])
                    print(f"✅ 加载权重: {orig_name} -> {custom_name}, shape: {original_weights[orig_name].shape}")
                    mapped_count += 1
                else:
                    print(f"❌ 警告: 权重形状不匹配 - {orig_name} vs {custom_name}, shape: {original_weights[orig_name].shape} vs {custom_state_dict[custom_name].shape}")
            else:
                print(f"❌ 警告: 未找到权重 - {orig_name}")
    
    print(f"权重映射完成: {mapped_count}/{total_count} 个权重已映射")
    
    custom_model.load_state_dict(custom_state_dict)
    return custom_model


# 4. 采样函数
def sample_tokens(logits, temperature=1.0, top_k=0, top_p=1.0):
    """从logits中采样tokens，支持温度调节、top-k和top-p采样
    
    Args:
        logits: 模型的原始输出logits [..., vocab_size]
        temperature: 温度参数(>0)，值越小越确定，越大越随机
        top_k: 仅从概率最高的k个token中采样(0表示禁用)
        top_p: nucleus采样参数(0-1)，从累积概率超过p的最小token集合中采样
    
    Returns:
        采样得到的token索引 [..., 1]
    """
    # 防止除零错误
    temperature = max(temperature, 1e-5)
    
    # 应用温度调节
    if temperature != 1.0:
        logits = logits / temperature
    
    # 计算概率分布
    probs = torch.softmax(logits, dim=-1)
    
    # top-k过滤
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))  # 确保不超过词汇表大小
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
    
    # top-p (nucleus) 过滤
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 创建mask：移除累积概率超过p的token，但要保留第一个超出阈值的位置
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        
        # 将masked位置的概率设为0
        sorted_probs[mask] = 0.0
        
        # 重新归一化概率
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        
        # 恢复原始顺序
        probs = sorted_probs.gather(-1, sorted_indices.argsort(-1))
    
    # 从处理后的分布中采样
    return torch.multinomial(probs, num_samples=1)

import time

def generate_tokens_autoregressive(model, tokenizer, input_ids, max_tokens=50, temperature=0.0, top_k=0, top_p=1.0):
    """自回归生成tokens, 并测量每个token生成时间"""
    model.eval()
    generated_tokens = input_ids.clone()
    total_time = 0.0
    token_times = []
    
    with torch.no_grad():
        for i in range(max_tokens):
            start_time = time.time()
            
            # 获取logits
            logits = model(generated_tokens)  # [batch_size, vocab_size]
            
            next_token = sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            
            # 添加到生成的序列中
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
            # 记录单个token生成时间
            elapsed = time.time() - start_time
            token_times.append(elapsed)
            total_time += elapsed
            
            print(f"Token {i+1} 生成时间: {elapsed:.4f}秒")
            
            # 检查是否生成了结束token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 输出统计信息
    print("\n生成统计:")
    print(f"总生成token数: {len(token_times)}")
    print(f"总耗时: {total_time:.4f}秒")
    print(f"平均每个token生成时间: {total_time/len(token_times):.4f}秒")
    if len(token_times) > 1:
        print(f"最快生成时间: {min(token_times):.4f}秒")
        print(f"最慢生成时间: {max(token_times):.4f}秒")
    
    return generated_tokens

def decode_tokens_to_text(tokenizer, tokens):
    """将tokens解码为文本"""
    # 移除batch维度
    if tokens.dim() > 1:
        tokens = tokens[0]
    
    # 解码为文本
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

# 5. 主函数：加载模型并测试
def main():
    # 加载配置
    config = load_llama_config("models/TinyStories-656K/config.json")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/TinyStories-656K/")
    print(f"Tokenizer词汇表大小: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # 创建自定义模型
    custom_model = CustomLlamaModel(config)
    print("custom model:", custom_model.state_dict().keys())
    # 加载原始权重
    original_weights = load_llama_weights("models/TinyStories-656K/model.safetensors")
    
    # 映射权重到自定义模型
    custom_model = map_weights_to_custom_model(custom_model, original_weights)
    custom_model.eval()  # 切换到评估模式
    
    print("自定义模型权重加载完成!")
    
    # 测试自回归生成
    with torch.no_grad():
        # # 准备输入文本
        input_text = "One day, a girl named Lily went for a "
        print(f"\n输入文本: '{input_text}'")
        
        # # 编码输入文本
        input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True)
        print(f"输入tokens: {input_ids[0].tolist()}")
        
        # # 调试：比较与main.py的差异
        # print(f"main.py的输入tokens: [1, 80, 429, 1168, 303, 1444]")
        # print(f"我们的输入tokens: {input_ids[0].tolist()}")
        
        # 检查tokenizer是否正确
        # decoded_back = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # print(f"解码回文本: '{decoded_back}'")
        
        # 测试logits输出
        # print("\n=== 测试logits输出 ===")
        # logits = custom_model(input_ids)
        # print("logits:", logits.reshape(-1,)[64:128])

        # # 自回归生成
        print("\n开始自回归生成...")
        generated_tokens = generate_tokens_autoregressive(
            custom_model, 
            tokenizer, 
            input_ids,
            temperature=1.0,
            max_tokens=100,
            top_k=1,
            top_p=1.0
        )
        
        print('generated_tokens:', generated_tokens)
        # # 解码生成的文本
        generated_text = decode_tokens_to_text(tokenizer, generated_tokens)
        
        print(f"\n生成的tokens: {generated_tokens[0].tolist()}")
        print(f"生成的文本: '{generated_text}'")
        
        # # 显示生成过程
        # print("\n生成过程:")
        # current_tokens = input_ids.clone()
        # for i in range(min(10, len(generated_tokens[0]) - len(input_ids[0]))):
        #     current_text = decode_tokens_to_text(tokenizer, current_tokens)
        #     print(f"步骤 {i+1}: '{current_text}'")
            
        #     # 获取下一个token
        #     logits = custom_model(current_tokens)
        #     next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #     current_tokens = torch.cat([current_tokens, next_token], dim=-1)

if __name__ == "__main__":
    main()
    