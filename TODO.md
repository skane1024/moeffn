# moe_ffn reference 实现

## 任务概述
生成一个 moe_ffn reference 的实现代码：给定两个输入（二进制 .bin 文件）和对应的专家信息，得到一个输出。
- 第一个输入：token 输入，NP128 格式，可能是 int8，也可能是 float16
- 第二个输入：专家信息（router 矩阵），始终为 float16
- 一个专家，可以是一个 ONNX 模型，也可以给定两个权重数据以及量化参数
  - ONNX 模型：加载模型并调用推理接口
  - 权重 + 量化参数：根据量化参数进行量化/反量化计算，最后得到 int/float16 的输出
- 除数据文件外，还需要给定两组量化参数：
  - moe-ffn 整体的输入/输出量化参数
  - 专家的量化参数（参考 expert/expert_1.onnx 与 expert/quant_params.json）

## 输入定义

### 1. token 输入（.bin）
- 布局为 NP128，生成方式见 common.py 的 to_np128()：
  - (num_tokens, 3072) -> (3072, num_tokens) -> (24, 128, num_tokens) -> (24, num_tokens, 128)
  - num_tokens == 1 时 token 维 pad 到 2（补零），读取/写回都要处理这个 padding
- dtype：
  - int8：每元素 1 字节，参考 test_data*/token_data_np128_int8.bin
  - float16：每元素 2 字节，参考 test_data*/token_data_np128_f16.bin
- int8 时需用 moe-ffn 输入量化参数反量化为 float 后再参与计算：x = (q - zp) * scale

### 2. 专家信息（.bin）
- router 矩阵，形状 (NUM_EXPERTS=116, num_tokens)，float16，行优先裸数据
- 参考 test_data*/expert_tokens.bin（gen_golden.py 直接 tofile 写出）
- 非零值 = 该 token 路由到该专家的权重；每个 token 的 TOP_K=8 个专家权重之和 = 1.0
- 用 common.py 的 scan_expert_tokens(router, expert_id) 得到 (token_ids, weights)

## 量化参数

### moe-ffn 输入/输出量化参数
- per-tensor 格式：{"dtype": "int8", "scale": ..., "zeropoint": ...}
- 输入参数示例（gen_golden.py 的 IN_SCALE/IN_ZEROPOINT）：scale=0.047759633511304855, zeropoint=3
- 输出参数由 golden 生成时 int8_quantize 统计得到（运行日志打印的 out scale_zeropoint）
- 接口上可以 JSON 文件路径或直接 dict 传入

### 专家量化参数（参考 expert/quant_params.json）
- "input" / "output"：激活 per-tensor，{dtype, scale, zeropoint}
- "conv1.weight"：权重 per-channel，scales/zeropoints 各 640 个（按输出通道）
- "conv3.weight"：权重 per-channel，scales/zeropoints 各 3072 个（按输出通道）
- "conv_1_out"、"5"~"9"：中间张量 per-tensor，reference 按 QDQ 图逐节点使用（Slice 后 "5"/"8" 也需单独 QDQ，以 w72 模型为准）
- 量化/反量化公式与 common.py 保持一致：
  - 量化：q = clip(round(x / scale + zp), -128, 127)
  - 反量化：x = (q - zp) * scale；权重按通道广播 scale/zeropoint

## 专家的两种形式

### 形式 A：ONNX 模型
- 参考 expert/expert_72_qdq.onnx：输入/输出均为 FLOAT (1, 3072, 1, 72)（w1 版为 (1, 3072, 1, 1)）
- 模型结构：Conv1x1(3072->640) -> Slice 拆分(320/320) -> SiLU(gate) * up -> Conv1x1(320->3072)
- 调用方式见 moe_ffn_ref.py 的 expert_onnx_forward()：
  - (num_tokens, 3072) -> transpose -> (1, 3072, 1, max_tokens)，float32 送入 session
  - 输出 transpose 回 (num_tokens, 3072)
  - prefill 阶段每个专家分到的 token 数可能超过模型 token 维上限（w72=72），按上限分块推理，最后一块补零送入、输出丢弃补零部分（专家 FFN 无跨 token 计算，分块与整批逐 token 等价）
  - session 按模型路径缓存，避免 116 个专家重复加载
- 注意：与硬件对齐时应使用 QDQ 模型（expert_72_qdq.onnx），fp32 模型误差偏大（见 readme.md 坑 1）

### 形式 B：两个权重数据 + 量化参数
- w1 = conv1.weight [640, 3072]，w3 = conv3.weight [3072, 320]（1x1 conv 等价矩阵乘）
- 权重可为 int8（配 per-channel scales/zeropoints）或 float（expert/weights/*.npy，由 numpy_infer.py 提取）
- 计算流程（逐节点模拟 QDQ 图，以 w72 模型为准）：
  1. 专家输入按 "input" 量化/反量化
  2. w1 / w3 按 per-channel 参数量化/反量化
  3. hidden = x @ w1.T，按 "conv_1_out" 量化/反量化
  4. Slice 拆分 gate/up，分别按 "5"/"8" 量化/反量化
  5. sigmoid(gate) 按 "6"、gate*sigmoid 按 "7"、silu*up 按 "9" 量化/反量化
  6. out = swiglu @ w3.T，按 "output" 量化/反量化后返回 float32
- 中间张量 QDQ（含 "5"/"8"）一个都不能省，否则与 QDQ 模型误差 ~0.03~0.1

## 接口设计（已实现于 moe_ffn_ref.py）

reference 函数直接接收数据数组，.bin/.npy/json 文件由调用方先读取：

```python
def moe_ffn_reference(
    token_np128,      # ndarray: NP128 token 输入数据 (int8 或 float16)
    router_matrix,    # ndarray: router 矩阵 (116, num_tokens) fp16
    num_tokens,       # int: token 数
    in_scale,         # float: moe-ffn 输入量化 scale (token_dtype=="int8" 时反量化用)
    in_zeropoint,     # int:   moe-ffn 输入量化 zeropoint
    out_scale,        # float: moe-ffn 输出量化 scale (out_dtype=="int8" 时量化用)
    out_zeropoint,    # int:   moe-ffn 输出量化 zeropoint
    expert,           # dict: 专家描述, 见下方两种形式
    token_dtype,      # "int8" | "float16": token 输入数据类型
    out_dtype,        # "int8" | "float16": 输出类型
):
    # 返回 NP128 输出数组 (24, max(num_tokens,2), 128)
```

expert 参数的两种取值：

```python
# 形式 A: ONNX 模型 (推理接口本身需要模型路径)
{"mode": "onnx", "path": "expert/expert_1.onnx"}

# 形式 B: 权重数据 + 量化参数 (均为调用方读好的数据)
{"mode": "weights",
 "w1": np.load("expert/weights/conv1_weight.npy"),   # ndarray
 "w3": np.load("expert/weights/conv3_weight.npy"),   # ndarray
 "quant_params": json.load(open("expert/quant_params.json"))}  # dict
```

## 整体计算流程
1. 读 token .bin -> reshape 为 NP128 (24, num_tokens, 128) -> 还原为 (num_tokens, 3072)；int8 则按 moe_in_quant 反量化为 float
2. 读 expert .bin -> router 矩阵 (116, num_tokens) fp16
3. golden = zeros (num_tokens, 3072)，float32 累加
4. 对每个 expert_id in range(116)：
   - token_ids, weights = scan_expert_tokens(router, expert_id)；为空则跳过
   - expert_out = 专家计算(token_data[token_ids])，按 expert 的 mode 走形式 A 或形式 B
   - golden[token_ids] += weights[:, None] * expert_out
5. golden -> to_np128 -> 按 out_dtype 输出：float16 直接写 / int8 按 moe_out_quant 量化
6. 与 golden_result_np128_f16.bin / golden_result_np128_int8.bin 对比验证（参考 compare*.py）

## 注意点
- 1-token（generation 阶段）NP128 pad 到 2 行，读取与写回要保持一致
- w1 专家模型 token 维固定为 1；多 token 用 w72 模型或动态轴版本
- 加权累加用 float32，最后再转目标精度，避免 fp16 累加误差
- 多专家共用同一份权重/量化参数时，接口需支持按 expert_id 区分（后续可扩展为 dict[int, expert]）
