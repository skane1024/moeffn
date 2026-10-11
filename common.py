import numpy as np

N_DIMS       = 3072             # 每 token 特征维度
NUM_EXPERTS  = 116              # 专家的个数
TOP_K        = 8                # 每个 token 被路由到的专家数


# int8 量化值域
INT8_QMIN = np.iinfo(np.int8).min   # -128
INT8_QMAX = np.iinfo(np.int8).max   # 127


# 计算量化参数
# symmetric=True  : 对称量化, scale = max(|x_min|,|x_max|) / QMAX, zero_point = 0
# symmetric=False : 非对称量化, scale 由 x_min/x_max 的完整范围均分到 [QMIN, QMAX]
def compute_quant_params(data, symmetric=True):
    x_min = float(data.min())
    x_max = float(data.max())
    if symmetric:
        scale = max(abs(x_min), abs(x_max)) / INT8_QMAX
        zero_point = 0
    else:
        scale = (x_max - x_min) / (INT8_QMAX - INT8_QMIN)
        zero_point = INT8_QMIN - int(np.round(x_min / scale))
        zero_point = int(np.clip(zero_point, INT8_QMIN, INT8_QMAX))

    if scale <= 0.0:
        scale = 1.0
    return scale, zero_point


# 对数据做 int8 量化
# 给定 scale 时按给定参数量化；未给定时统计数据范围自动计算量化参数
# symmetric 仅影响自动计算参数的行为，对称/非对称可选
def int8_quantize(data, scale=None, zero_point=None, symmetric=True):
    x = np.asarray(data, dtype=np.float32)
    if scale is None:
        scale, zero_point = compute_quant_params(x, symmetric)
    elif zero_point is None:
        zero_point = 0

    q = np.round(x / scale + zero_point)
    q = np.clip(q, INT8_QMIN, INT8_QMAX).astype(np.int8)
    return q, scale, zero_point


def to_np128(data, num_tokens):
    data = data.reshape((num_tokens, N_DIMS))
    data = data.transpose((1, 0))
    data = data.reshape(N_DIMS // 128, 128, num_tokens)
    data = data.transpose(0, 2, 1)
    if data.shape[1] == 1: #for generate golden result, 1 pad to 2
        data = np.pad(data, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)
    return data


# 从 router 矩阵扫描当前专家的 (token_id, weight)
def scan_expert_tokens(router_matrix, expert_id):
    expert_row = router_matrix[expert_id]
    token_ids  = np.nonzero(expert_row)[0]
    weights    = expert_row[token_ids]
    return token_ids, weights


# 专家计算
def expert_forward(input_data, num_tokens, expert_onnx_model=""):
    # print(input_data.shape)
    if expert_onnx_model != "":
        import onnxruntime as ort
        input_data_1 = input_data.transpose((1, 0))
        input_data_1 = input_data_1.reshape((1, N_DIMS, 1, num_tokens))
        input_data_1 = input_data_1.astype(np.float32)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(expert_onnx_model, so)
        input_name_0 = session.get_inputs()[0].name
        input_data_map = {input_name_0: input_data_1}
        outputs = session.run(None, input_data_map)[0]
        outputs = outputs.transpose(0, 3, 2, 1)
        outputs = outputs.reshape(num_tokens, N_DIMS)
        print("expert result: ", np.min(outputs), np.max(outputs))
        return outputs
    return input_data


# 生成 router 矩阵 (NUM_EXPERTS, num_tokens), 非零值 = 权重
#每个 token 只被路由到 8 个专家" 每个 token 分给它所选 8 个专家的权重之和 = 1.0
def build_router_matrix(rng, num_tokens):
    router_matrix = np.zeros((NUM_EXPERTS, num_tokens), dtype=np.float32)
    routing_scores = rng.random((NUM_EXPERTS, num_tokens), dtype=np.float32)
    topk_experts = np.argpartition(-routing_scores, TOP_K, axis=0)[:TOP_K]
    for token_id in range(num_tokens):
        selected_experts = topk_experts[:, token_id]
        raw_weights = np.exp(routing_scores[selected_experts, token_id])
        router_matrix[selected_experts, token_id] = raw_weights / raw_weights.sum()
    router_matrix = router_matrix.astype(np.float16)
    return router_matrix
