import json

import numpy as np

from common import INT8_QMIN, INT8_QMAX, to_np128, int8_quantize, scan_expert_tokens


# 量化参数: 支持直接传 dict, 或传 json 文件路径
def load_quant_params(params):
    if isinstance(params, str):
        with open(params, "r") as f:
            return json.load(f)
    return params


# 单值量化参数 (per-tensor): {dtype, scale, zeropoint}
def get_scale_zeropoint(param):
    return float(param["scale"]), int(param["zeropoint"])


# int8 反量化: x = (q - zp) * scale
def int8_dequantize(q_data, scale, zeropoint):
    return (np.asarray(q_data, dtype=np.float32) - zeropoint) * scale


# 从 NP128 还原为 (num_tokens, n_dims)
# NP128 布局: (n_dims // 128, num_tokens, 128), num_tokens == 1 时 token 维 pad 到 2
def from_np128(data, num_tokens, n_dims):
    data = np.asarray(data)
    data = data.reshape(n_dims // 128, -1, 128)
    data = data[:, :num_tokens, :]
    data = data.transpose(0, 2, 1)               # (n_dims//128, 128, num_tokens)
    data = data.reshape(n_dims, num_tokens)
    return data.T                                 # (num_tokens, n_dims)


# 解析 NP128 的 token 输入数据, 返回 float32 的 (num_tokens, n_dims)
# token_dtype == "int8" 时按 moe-ffn 输入量化参数反量化
def load_token_input(token_np128, num_tokens, n_dims, token_dtype, in_scale, in_zeropoint):
    if token_dtype == "int8":
        np128 = np.asarray(token_np128).reshape(n_dims // 128, max(num_tokens, 2), 128)
        return int8_dequantize(from_np128(np128, num_tokens, n_dims), in_scale, in_zeropoint)
    if token_dtype == "float16":
        np128 = np.asarray(token_np128).reshape(n_dims // 128, max(num_tokens, 2), 128)
        return from_np128(np128, num_tokens, n_dims).astype(np.float32)
    raise ValueError(f"unsupported token dtype: {token_dtype}")


# ONNX session 缓存, 避免每个专家重复加载模型
_ort_sessions = {}


def get_ort_session(model_path):
    if model_path not in _ort_sessions:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _ort_sessions[model_path] = ort.InferenceSession(model_path, so)
    return _ort_sessions[model_path]


# 专家形式 A: ONNX 模型推理
# 模型输入/输出形状 (1, n_dims, 1, max_tokens), float32
# 模型 token 维固定 (w1/w72), token 数超过上限时分块推理;
# 最后一块不足 max_tokens 时补零送入, 推理后丢弃补零部分
# (专家 FFN 无跨 token 计算, 分块与整批结果逐 token 一致)
def expert_onnx_forward(input_data, num_tokens, n_dims, model_path):
    x = np.asarray(input_data, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    session = get_ort_session(model_path)
    input_name = session.get_inputs()[0].name
    max_tokens = session.get_inputs()[0].shape[3]
    if not isinstance(max_tokens, int):     # 动态轴模型, 一次送入
        max_tokens = num_tokens

    results = []
    for start in range(0, num_tokens, max_tokens):
        chunk = x[start:start + max_tokens]
        n = chunk.shape[0]
        if n < max_tokens:
            pad = np.zeros((max_tokens, n_dims), dtype=np.float32)
            pad[:n] = chunk
            chunk = pad
        inp = chunk.T.reshape(1, n_dims, 1, max_tokens)
        out = session.run(None, {input_name: inp})[0]
        out = out.transpose(0, 3, 2, 1).reshape(max_tokens, n_dims)
        results.append(out[:n])
    return np.concatenate(results, axis=0)


# 权重按 per-channel 参数模拟量化: 量化后再反量化回 float
# scales/zeropoints 沿权重第 0 维 (输出通道) 广播
def quantize_weight_dequant(w_float, scales, zeropoints):
    w = np.asarray(w_float, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32).reshape(-1, *([1] * (w.ndim - 1)))
    z = np.asarray(zeropoints, dtype=np.float32).reshape(-1, *([1] * (w.ndim - 1)))
    q = np.clip(np.round(w / s + z), INT8_QMIN, INT8_QMAX)
    return (q - z) * s


# 按 named 参数对一个张量做量化/反量化 (模拟 QDQ 节点), 参数缺失则原样返回
def qdq_tensor(x, params, name):
    if name not in params:
        return x
    s, z = get_scale_zeropoint(params[name])
    q, _, _ = int8_quantize(x, s, z)
    return int8_dequantize(q, s, z)


# 专家形式 B: 两个权重数据 + 量化参数
# 计算流程模拟 QDQ 模型 (expert_1_qdq.onnx):
#   input QDQ -> Conv1x1 -> conv_1_out QDQ -> Slice 拆分
#   -> Sigmoid -> QDQ("6") -> gate*sigmoid QDQ("7") -> *up QDQ("9")
#   -> Conv1x1 -> output QDQ
# 权重按 per-channel 参数量化/反量化
# 返回 float32 的 (num_tokens, N_DIMS)
def expert_weights_forward(input_data, w1, w3, quant_params):
    params = load_quant_params(quant_params)
    x = np.asarray(input_data, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    x = qdq_tensor(x, params, "input")

    w1_dq = quantize_weight_dequant(w1, params["conv1.weight"]["scales"],
                                    params["conv1.weight"]["zeropoints"])
    w3_dq = quantize_weight_dequant(w3, params["conv3.weight"]["scales"],
                                    params["conv3.weight"]["zeropoints"])
    w1_2d = w1_dq.reshape(w1_dq.shape[0], -1)          # (640, 3072)
    w3_2d = w3_dq.reshape(w3_dq.shape[0], -1)          # (3072, 320)

    hidden = qdq_tensor(x @ w1_2d.T, params, "conv_1_out")   # (num_tokens, 640)
    gate_dims = hidden.shape[1] // 2
    # Slice 拆分后 gate/up 各自再过一次 QDQ ("5"/"8"), 与 w72 QDQ 模型一致
    gate = qdq_tensor(hidden[:, :gate_dims], params, "5")
    up = qdq_tensor(hidden[:, gate_dims:], params, "8")
    sigmoid = qdq_tensor(1.0 / (1.0 + np.exp(-gate)), params, "6")
    silu = qdq_tensor(gate * sigmoid, params, "7")      # SiLU(gate)
    swiglu = qdq_tensor(silu * up, params, "9")
    output = swiglu @ w3_2d.T                           # (num_tokens, 3072)

    return qdq_tensor(output, params, "output")


# 加载专家权重 (支持 .npy 路径或 ndarray)
def load_expert_weight(w):
    if isinstance(w, str):
        return np.load(w)
    return w


# 单个专家计算, 按 expert["mode"] 分发
def run_expert(input_data, num_tokens, n_dims, expert):
    mode = expert.get("mode", "onnx")
    if mode == "onnx":
        return expert_onnx_forward(input_data, num_tokens, n_dims, expert["path"])
    if mode == "weights":
        w1 = load_expert_weight(expert["w1"])
        w3 = load_expert_weight(expert["w3"])
        return expert_weights_forward(input_data, w1, w3, expert["quant_params"])
    raise ValueError(f"unsupported expert mode: {mode}")


# moe_ffn reference 主入口, 直接接收数据数组 (文件读取由调用方完成)
# token_np128:    NP128 token 输入数据, ndarray (int8 或 float16)
# router_matrix:  router 矩阵数据, ndarray (num_experts, num_tokens) fp16
# num_tokens / n_dims / num_experts: 均由外部传入, 不依赖全局常量
# in_scale / in_zeropoint:   moe-ffn 输入量化参数 (token_dtype == "int8" 时用于反量化)
# out_scale / out_zeropoint: moe-ffn 输出量化参数 (out_dtype == "int8" 时用于量化)
# expert:         {"mode": "onnx", "path": 模型路径} 或
#                 {"mode": "weights", "w1": ndarray, "w3": ndarray, "quant_params": dict/json 路径}
# 返回 NP128 输出数组 (n_dims//128, max(num_tokens,2), 128), dtype 由 out_dtype 决定
def moe_ffn_reference(token_np128, router_matrix, num_tokens, n_dims, num_experts,
                      in_scale, in_zeropoint, out_scale, out_zeropoint, expert,
                      token_dtype="int8", out_dtype="int8"):
    token_data = load_token_input(token_np128, num_tokens, n_dims, token_dtype, in_scale, in_zeropoint)
    router = np.asarray(router_matrix).reshape(num_experts, num_tokens)

    result = np.zeros((num_tokens, n_dims), dtype=np.float32)
    for expert_id in range(num_experts):
        token_ids, token_weights = scan_expert_tokens(router, expert_id)
        if len(token_ids) == 0:
            continue
        expert_in = token_data[token_ids]
        expert_out = run_expert(expert_in, len(token_ids), n_dims, expert)
        result[token_ids] += token_weights[:, None].astype(np.float32) * expert_out

    result_np128 = to_np128(result.astype(np.float16), num_tokens, n_dims)
    if out_dtype == "float16":
        return result_np128.astype(np.float16)
    if out_dtype == "int8":
        out_q = int8_quantize(result_np128, out_scale, out_zeropoint)[0]
        return out_q
    raise ValueError(f"unsupported out dtype: {out_dtype}")


if __name__ == "__main__":
    N_DIMS = 3072
    NUM_EXPERTS = 116
    NUM_TOKENS = 1
    in_scale, in_zeropoint = 0.047759633511304855, 3
    out_scale, out_zeropoint = 0.01187253937007874, 0
    expert_quant = json.load(open("expert/quant_params.json"))
    token_f16 = np.fromfile(f"test_data_generation/token_data_np128_f16.bin", dtype=np.float16)
    router = np.fromfile(f"test_data_generation/expert_tokens.bin", dtype=np.float16)
    w1 = np.load("expert/weights/conv1_weight.npy")
    w3 = np.load("expert/weights/conv3_weight.npy")

    # 形式 A: ONNX 模型 (QDQ), 同样输入对比 golden (golden 由 w72 模型生成)
    ref_onnx = moe_ffn_reference(
        token_f16, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "onnx", "path": "expert/expert_1_qdq.onnx"},
        token_dtype="float16", out_dtype="float16")
    golden_f16 = np.fromfile(f"test_data_generation/golden_result_np128_f16.bin", dtype=np.float16).reshape(N_DIMS // 128, 2, 128)
    diff_onnx = np.max(np.abs(ref_onnx.astype(np.float32) - golden_f16.astype(np.float32)))
    print("onnx mode vs golden f16 max diff:", diff_onnx)



    token_int8 = np.fromfile(f"test_data_generation/token_data_np128_int8.bin", dtype=np.int8)
    ref_int8 = moe_ffn_reference(
        token_int8, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "onnx", "path": "expert/expert_1_qdq.onnx"},
        token_dtype="int8", out_dtype="int8")
    golden_int8 = np.fromfile(f"test_data_generation/golden_result_np128_int8.bin", dtype=np.int8).reshape(N_DIMS // 128, 2, 128)
    diff_int8 = np.max(np.abs(ref_int8.astype(np.float32) - golden_int8.astype(np.float32)))
    print("int8 mode vs golden int8 max diff:", diff_int8)


    # 形式 B: 权重 + 量化参数, float16 输入 -> float16 输出
    ref_f16 = moe_ffn_reference(
        token_f16, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "weights", "w1": w1, "w3": w3, "quant_params": expert_quant},
        token_dtype="float16", out_dtype="float16")
    golden_f16 = np.fromfile(f"test_data_generation/golden_result_np128_f16.bin", dtype=np.float16).reshape(N_DIMS // 128, 2, 128)
    diff = np.max(np.abs(ref_f16.astype(np.float32) - golden_f16.astype(np.float32)))
    print("weights mode vs golden f16 max diff:", diff)


    token_int8 = np.fromfile(f"test_data_generation/token_data_np128_int8.bin", dtype=np.int8)
    ref_int8 = moe_ffn_reference(
        token_int8, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "weights", "w1": w1, "w3": w3, "quant_params": expert_quant},
        token_dtype="int8", out_dtype="int8")
    golden_int8 = np.fromfile(f"test_data_generation/golden_result_np128_int8.bin", dtype=np.int8).reshape(N_DIMS // 128, 2, 128)
    diff = np.max(np.abs(ref_int8.astype(np.float32) - golden_int8.astype(np.float32)))
    print("weights mode vs golden int8 max diff:", diff)



    NUM_TOKENS = 1024
    in_scale, in_zeropoint = 0.047759633511304855, 3
    out_scale, out_zeropoint = 0.022484005905511813, 0
    expert_quant = json.load(open("expert/quant_params.json"))
    token_f16 = np.fromfile(f"test_data/token_data_np128_f16.bin", dtype=np.float16)
    router = np.fromfile(f"test_data/expert_tokens.bin", dtype=np.float16)

    # 形式 A: ONNX 模型 (QDQ), 同样输入对比 golden
    ref_onnx = moe_ffn_reference(
        token_f16, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "onnx", "path": "expert/expert_72_qdq.onnx"},
        token_dtype="float16", out_dtype="float16")
    golden_f16 = np.fromfile(f"test_data/golden_result_np128_f16.bin", dtype=np.float16).reshape(N_DIMS // 128, NUM_TOKENS, 128)
    diff_onnx = np.max(np.abs(ref_onnx.astype(np.float32) - golden_f16.astype(np.float32)))
    print("onnx mode vs golden f16 max diff:", diff_onnx)


# 形式 A: ONNX 模型 (QDQ), 同样输入对比 golden
    ref_onnx = moe_ffn_reference(
        token_f16, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "weights", "w1": w1, "w3": w3, "quant_params": expert_quant},
        token_dtype="float16", out_dtype="float16")
    golden_f16 = np.fromfile(f"test_data/golden_result_np128_f16.bin", dtype=np.float16).reshape(N_DIMS // 128, NUM_TOKENS, 128)
    diff_onnx = np.max(np.abs(ref_onnx.astype(np.float32) - golden_f16.astype(np.float32)))
    print("weight mode vs golden f16 max diff:", diff_onnx)


    token_int8 = np.fromfile(f"test_data/token_data_np128_int8.bin", dtype=np.int8)
    golden_int8 = np.fromfile(f"test_data/golden_result_np128_int8.bin", dtype=np.int8).reshape(N_DIMS // 128, NUM_TOKENS, 128)
    ref_int8 = moe_ffn_reference(
        token_int8, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "weights", "w1": w1, "w3": w3, "quant_params": expert_quant},
        token_dtype="int8", out_dtype="int8")
    diff = np.max(np.abs(ref_int8.astype(np.float32) - golden_int8.astype(np.float32)))
    print("weights mode vs golden int8 max diff:", diff)


    ref_int8 = moe_ffn_reference(
        token_int8, router, NUM_TOKENS, N_DIMS, NUM_EXPERTS,
        in_scale, in_zeropoint, out_scale, out_zeropoint,
        {"mode": "onnx", "path": "expert/expert_72_qdq.onnx"},
        token_dtype="int8", out_dtype="int8")
    diff = np.max(np.abs(ref_int8.astype(np.float32) - golden_int8.astype(np.float32)))
    print("onnx mode vs golden int8 max diff:", diff)




