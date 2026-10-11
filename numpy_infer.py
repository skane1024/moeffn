import os
import numpy as np
import onnx
from onnx import numpy_helper

from common import N_DIMS

MODEL_PATH = "expert/expert_72.onnx"
WEIGHT_DIR = "expert/weights"

# 专家模型中间层维度 (SwiGLU 门控/上投影各占一半)
HIDDEN_DIMS = 640
GATE_DIMS   = HIDDEN_DIMS // 2


# 从 ONNX 模型中提取权重并保存为 .npy 文件
def extract_weights(model_path=MODEL_PATH, out_dir=WEIGHT_DIR):
    model = onnx.load(model_path)
    weights = {}
    os.makedirs(out_dir, exist_ok=True)
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        weights[init.name] = arr
        save_path = os.path.join(out_dir, init.name.replace(".", "_") + ".npy")
        np.save(save_path, arr)
        print(f"saved {init.name} shape={arr.shape} dtype={arr.dtype} -> {save_path}")
    return weights


# 用 numpy 模拟整个模型的计算过程
# 模型结构: Conv(1x1) -> Slice 拆分 -> SiLU(gate) * up -> Conv(1x1)
# 输入/输出形状: (num_tokens, N_DIMS)
def numpy_forward(input_data, w1, w3):
    # w1: [640, 3072, 1, 1] -> [640, 3072], 1x1 卷积等价于矩阵乘
    w1 = np.asarray(w1).reshape(HIDDEN_DIMS, N_DIMS)
    # w3: [3072, 320, 1, 1] -> [3072, 320]
    w3 = np.asarray(w3).reshape(N_DIMS, GATE_DIMS)

    x = np.asarray(input_data, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    hidden = x @ w1.T                       # (num_tokens, 640)
    gate = hidden[:, :GATE_DIMS]            # Slice [0:320]
    up   = hidden[:, GATE_DIMS:]            # Slice [320:end]
    activated = gate * (1.0 / (1.0 + np.exp(-gate)))   # SiLU = gate * sigmoid(gate)
    swiglu = activated * up                 # (num_tokens, 320)
    output = swiglu @ w3.T                  # (num_tokens, 3072)
    return output


# 从 .npy 文件加载之前保存的权重
def load_weights(weight_dir=WEIGHT_DIR):
    w1 = np.load(os.path.join(weight_dir, "conv1_weight.npy"))
    w3 = np.load(os.path.join(weight_dir, "conv3_weight.npy"))
    return w1, w3


if __name__ == "__main__":
    # 1. 提取并保存权重
    weights = extract_weights()

    # 2. 用 numpy 复现模型计算
    w1, w3 = load_weights()
    rng = np.random.default_rng(0)
    # w1 模型 token 维固定为 1
    num_tokens = 72
    input_data = rng.standard_normal(size=(num_tokens, N_DIMS)).astype(np.float32)
    np_out = numpy_forward(input_data, w1, w3)

    # 3. 与 onnxruntime 推理结果对比验证
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(MODEL_PATH, so)
    input_name = session.get_inputs()[0].name
    # ONNX 输入形状 (1, 3072, 1, num_tokens), 与 expert_forward 的排布一致
    ort_in = input_data.T.reshape(1, N_DIMS, 1, num_tokens)
    ort_out = session.run(None, {input_name: ort_in})[0]
    ort_out = ort_out.transpose(0, 3, 2, 1).reshape(num_tokens, N_DIMS)

    max_diff = np.max(np.abs(np_out - ort_out))
    print(f"numpy vs onnxruntime max diff: {max_diff}")
    assert max_diff < 1e-3, "numpy 结果与 onnxruntime 不一致!"
    print("验证通过")
