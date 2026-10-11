import torch
import torch.nn.functional as F

NUM_TOKENS   = 512          # token 总数
N_DIMS          = 1024         # 每 token 特征维度
PACK         = 128          # NP128: 每物理块 128 维
NUM_GROUPS   = N_DIMS // PACK  # = 8 个 group
NUM_EXPERTS  = 116
TOP_K        = 8           # 每个 token 被路由到的专家数
BATCH_TOKENS = 8            # VDSP 每批处理的 token 数
CLUSTER = 4                                 # 核数


torch.manual_seed(42)
NUM_TOKENS, N_DIMS, ffn_hiddeN_DIMS = 512, 1024, 640          # token数, 特征, 中间维度

NUM_EXPERTS = 8
TOP_K = 2

# ---- 输入 ----
x = torch.randn(NUM_TOKENS, N_DIMS) / 10

# ---- MoE 路由器 ----
router = torch.randn(N_DIMS, NUM_EXPERTS)

# ---- 4 个 Expert，每个 Expert 与原 Gated FFN 结构相同 (up, gate, down) ----
experts = []
for _ in range(NUM_EXPERTS):
    up = torch.randn(N_DIMS, ffn_hiddeN_DIMS) / 10
    gate = torch.randn(N_DIMS, ffn_hiddeN_DIMS) / 10
    down = torch.randn(ffn_hiddeN_DIMS, N_DIMS) / 10
    experts.append((up, gate, down))
    
    
def moe_step_by_step(x, router, experts, TOP_K=2):
    """
      1. 为每个 token 计算路由分数logits
      2. 选出分数最高的 TOP_K 个专家，用这些分数作为权重
      3. 对每个 token,将选中的各个专家的输出加权求和
    """
    NUM_TOKENS, N_DIMS = x.shape
    H_out = experts[0][2].shape[1]          # down 矩阵的输出维度，等于 H

    # 初始化输出，全 0
    y = torch.zeros(NUM_TOKENS, H_out)

    # ----- 路由阶段：一次算出所有 token 的 logits -----
    logits = x @ router                     # (NUM_TOKENS, NUM_EXPERTS)
    # 对每个 token，取出 TOP_K 个最大 logits 及其对应的专家编号
    topk_prob, topk_ids = torch.topk(logits, TOP_K, dim=-1)  
    # topk_prob: (NUM_TOKENS, TOP_K)   每个 token 被选中专家的原始 logits（作为权重）
    # topk_ids : (NUM_TOKENS, TOP_K)   对应的专家编号

    # ----- 逐 token 计算加权输出 -----
    for t in range(NUM_TOKENS):               # 遍历每一个 token
        token_vec = x[t]                   # 当前 token 的向量 (H,)
        for k in range(TOP_K):             # 遍历该 token 选中的 TOP_K 个专家
            expert_id = topk_ids[t, k].item()   # 专家编号  item 张量转换为 Python 标准的数字
            weight = topk_prob[t, k]       # 该专家的权重（logits 值）

            up, gate, down = experts[expert_id] # 取出这个专家的参数

            # ----- 标准的 Gated FFN -----
            gate_out = F.gelu(token_vec @ gate)   # (FFN_HIDDEN_DIM,)
            up_out   = token_vec @ up            # (FFN_HIDDEN_DIM,)
            expert_out = (gate_out * up_out) @ down  # (H_out,)

            # 加权并累加到该 token 的最终输出上
            y[t] += weight * expert_out

    return y


# ========== MoE 基准实现 ==========
# MoE 核心思想：每个 token 选择一部分专家处理，加权求和。
def moe_baseline(x, router, experts, TOP_K=2):
    NUM_TOKENS = x.shape[0]
    H_out = experts[0][2].shape[1]  # down 输出维度 N_DIMS
    y = torch.zeros(NUM_TOKENS, H_out)

    # 路由 logits 与选中专家
    logits = x @ router                        # (NUM_TOKENS, NUM_EXPERTS)
    topk_prob, topk_ids = torch.topk(logits, TOP_K, dim=-1)  # (NUM_TOKENS, TOP_K)
    
    # 2. 外层循环：遍历每一个专家
    for e, (up, gate, down) in enumerate(experts):
        # ----- 第一步：用“遍历访问”找出所有属于该专家的 token -----
        token_indices = []    # 存放 Token 在序列中的位置（行号）
        token_weights = []    # 存放这些 Token 分配给该专家的权重

        for i in range(NUM_TOKENS):
            # 遍历该 Token 的 TOP_K 个选择（通常 TOP_K=2，所以最多循环 2 次）
            for k in range(TOP_K):
                if topk_ids[i, k] == e:          # 如果该 Token 在第 k 位选择了专家 e
                    token_indices.append(i)       # 记录下这个 Token 的序号
                    token_weights.append(topk_prob[i, k])  # 取出对应的权重
                    break  # 一个 Token 在 TOP_K 中不可能重复选同一个专家，找到后直接跳出内层循环

        if len(token_indices) == 0:
            continue

        # 将 Python 列表转换为 PyTorch 张量，方便后续索引取值
        idx = torch.tensor(token_indices)
        w = torch.tensor(token_weights).unsqueeze(-1)  # 变成列向量 (selected, 1)

        # ----- 将所有 Token 集合成一批，整批送入专家网络计算 -----
        x_e = x[idx]
        z_gate = F.gelu(x_e @ gate)
        z_up   = x_e @ up
        out_e  = (z_gate * z_up) @ down   # (selected, H_out)

        # ----- 加权累加到最终结果 -----
        y[idx] += out_e * w
    return y




# 对专家的权重做切分，在 4个CLUSTER独立计算后合并
def moe_tensor_splite(x, router, experts,CLUSTER, TOP_K=2):
    NUM_TOKENS = x.shape[0]
    H_out = experts[0][2].shape[1]  # down 输出维度 N_DIMS
    
    # 路由 logits 与选中专家
    logits = x @ router                        # (NUM_TOKENS, NUM_EXPERTS)
    topk_prob, topk_ids = torch.topk(logits, TOP_K, dim=-1)  # (NUM_TOKENS, TOP_K)
    y = torch.zeros(NUM_TOKENS, H_out)
    
    for ci in range(CLUSTER):
        y_ci = torch.zeros(NUM_TOKENS, H_out)
        # 2. 外层循环：遍历每一个专家
        for e, (up, gate, down) in enumerate(experts):
            # ----- 第一步：用“遍历访问”找出所有属于该专家的 token -----
            token_indices = []    # 存放 Token 在序列中的位置（行号）
            token_weights = []    # 存放这些 Token 分配给该专家的权重

            for i in range(NUM_TOKENS):
                # 遍历该 Token 的 TOP_K 个选择（通常 TOP_K=2，所以最多循环 2 次）
                for k in range(TOP_K):
                    if topk_ids[i, k] == e:          # 如果该 Token 在第 k 位选择了专家 e
                        token_indices.append(i)       # 记录下这个 Token 的序号
                        token_weights.append(topk_prob[i, k])  # 取出对应的权重
                        break  # 一个 Token 在 TOP_K 中不可能重复选同一个专家，找到后直接跳出内层循环

            if len(token_indices) == 0:
                continue

            # 将 Python 列表转换为 PyTorch 张量，方便后续索引取值
            idx = torch.tensor(token_indices)
            w = torch.tensor(token_weights).unsqueeze(-1)  # 变成列向量 (selected, 1)

            # ----- 将所有 Token 集合成一批，整批送入专家网络计算 -----
            ups = torch.chunk(up, CLUSTER, dim=1)
            gates = torch.chunk(gate, CLUSTER, dim=1)
            downs = torch.chunk(down, CLUSTER, dim=0)
            
            x_e = x[idx]
            z_gate = F.gelu(x_e @ gates[ci])
            z_up   = x_e @ ups[ci]
            out_e  = (z_gate * z_up) @ downs[ci]   # (selected, H_out)

            # ----- 加权累加到最终结果 -----
            y_ci[idx] += out_e * w
        
        y.add_(y_ci)
    return y



# ========== 数值验证 ==========
y_base = moe_baseline(x, router, experts, TOP_K)
y_step = moe_step_by_step(x, router, experts, TOP_K)
y_split = moe_tensor_splite(x, router, experts,CLUSTER, TOP_K)

print("MoE baseline 输出 (前 3 行):")
print(y_base[:3])

print("\nmoe_step_by_step切分 输出 (前 3 行):")
print(y_step[:3])


print("\n张量切分 输出 (前 3 行):")
print(y_split[:3])

print("\nToken 切分 最大绝对误差:", (y_step - y_base).abs().max().item())

print("\n张量 切分 最大绝对误差:", (y_split - y_base).abs().max().item())