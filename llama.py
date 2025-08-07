import json
import torch
from safetensors.torch import load_file
from typing import Dict, Any, Tuple

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
        batch_size, num_heads, seq_len, head_dim = x.shape
        device = x.device

        # 生成频率（仅计算前半部分）
        theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device)[:head_dim//2] / head_dim))
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
        # 自定义注意力计算逻辑
        batch_size, seq_len, _ = x.shape
        
        # 自定义实现的投影操作
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        
        print("q=", q[0], "k=", k[0], "v=", v[0])
        q, k = self._rope(q), self._rope(k)

        print("q.shape:", q.shape, "k.shape:", k.shape, "v.shape:", v.shape)
        # 自定义注意力计算（示例）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = ACTFN_MAP[config["hidden_act"]]()
        self.gate_proj = torch.nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = torch.nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = torch.nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class CustomLlamaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]

        # 嵌入层
        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        
        # 自定义实现的Transformer层
        self.layers = torch.nn.ModuleList([
            self.create_decoder_layer(config) for _ in range(config["num_hidden_layers"])
        ])
        
    #     # 输出层
    #     self.norm = torch.nn.LayerNorm(self.hidden_size)

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
            "input_layernorm": torch.nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
            "post_attention_layernorm": torch.nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        })

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        
        # print("x=", x)

        for layer in self.layers:
            residual = x
            x = layer["input_layernorm"](x)
            x = layer["self_attn"](x)
            x = residual + x
            
            residual = x
            x = layer["post_attention_layernorm"](x)
            x = layer["mlp"](x)
            x = residual + x
            
        # x = self.norm(x)
        return x


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
    for custom_name in custom_state_dict.keys():
        orig_name = "model." + custom_name
        if orig_name in original_weights:
            # 确保权重形状匹配
            if original_weights[orig_name].shape == custom_state_dict[custom_name].shape:
                custom_state_dict[custom_name].copy_(original_weights[orig_name])
                print(f"加载权重: {orig_name} -> {custom_name}, shape: {original_weights[orig_name].shape}")
            else:
                print(f"警告: 权重形状不匹配 - {orig_name} vs {custom_name}, shape: {original_weights[orig_name].shape} vs {custom_state_dict[custom_name].shape}")
        else:
            print(f"警告: 未找到权重 - {orig_name}")
    
    custom_model.load_state_dict(custom_state_dict)
    return custom_model


# 4. 主函数：加载模型并测试
def main():
    # 加载配置
    config = load_llama_config("models/TinyStories-656K/config.json")
    
    # 创建自定义模型
    custom_model = CustomLlamaModel(config)
    print("custom model:", custom_model.state_dict().keys())
    # 加载原始权重
    original_weights = load_llama_weights("models/TinyStories-656K/model.safetensors")
    
    # 映射权重到自定义模型
    custom_model = map_weights_to_custom_model(custom_model, original_weights)
    custom_model.eval()  # 切换到评估模式
    
    print("自定义模型权重加载完成!")
    
    # 简单测试
    with torch.no_grad():
        input_ids = torch.tensor([[1, 80, 425]])  # 示例输入
        output = custom_model(input_ids)
        print(f"模型输出形状: {output.shape}")

if __name__ == "__main__":
    main()
    