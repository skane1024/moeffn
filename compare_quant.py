import os
import json

import numpy as np
import onnxruntime as ort

from common import N_DIMS, int8_quantize, to_np128
from compare import compare_arrays


FLOAT_MODEL = "expert/expert_1.onnx"
QDQ_MODEL   = "expert/expert_1_qdq.onnx"
PARAMS_JSON = "expert/quant_params.json"
OUT_DIR     = "quant_compare"


# 加载模型并用给定 fp32 输入推理, 返回 (num_tokens, N_DIMS) 的 fp32 输出
def run_model(model_path, input_name, input_feed):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(model_path, so)
    out = session.run(None, {input_name: input_feed})[0]
    # 模型输出布局 (1, N_DIMS, 1, num_tokens) -> (num_tokens, N_DIMS)
    num_tokens = input_feed.shape[-1]
    out = out.transpose(0, 3, 2, 1).reshape(num_tokens, N_DIMS)
    return out.astype(np.float32)


def main():
    NUM_TOKENS = 1

    # 1. 读取输入/输出量化参数
    params = json.load(open(PARAMS_JSON))
    in_scale, in_zp   = params["input"]["scale"],  params["input"]["zeropoint"]
    out_scale, out_zp = params["output"]["scale"], params["output"]["zeropoint"]
    print("in  scale/zp:", in_scale, in_zp)
    print("out scale/zp:", out_scale, out_zp)

    # 2. 生成 fp16 输入 token 数据
    rng = np.random.default_rng(42)
    token_data = rng.standard_normal(size=(NUM_TOKENS, N_DIMS)).astype(np.float16)

    # 输入 int8 量化数据 (np128 布局), 保存供参考
    token_data_np128 = to_np128(token_data, NUM_TOKENS)
    token_int8, _, _ = int8_quantize(token_data_np128, in_scale, in_zp)

    # 模型输入布局 (1, N_DIMS, 1, num_tokens), fp32
    model_in = token_data.astype(np.float32).transpose(1, 0).reshape(1, N_DIMS, 1, NUM_TOKENS)

    # 取模型真实输入名
    so = ort.SessionOptions()
    so.log_severity_level = 3
    input_name = ort.InferenceSession(FLOAT_MODEL, so).get_inputs()[0].name

    # 3. 浮点模型 vs QDQ 模型 (相同 fp32 输入)
    float_out = run_model(FLOAT_MODEL, input_name, model_in)
    qdq_out   = run_model(QDQ_MODEL,   input_name, model_in)

    # 4. 两者输出按相同 out_scale/out_zp 量化为 int8
    float_int8, _, _ = int8_quantize(float_out, out_scale, out_zp)
    qdq_int8,   _, _ = int8_quantize(qdq_out,   out_scale, out_zp)

    # 5. 精度对比
    print("\n===== 浮点输出 vs QDQ 输出 (fp32) =====")
    compare_arrays(float_out, qdq_out)
    print("\n===== 量化 int8 输出对比 =====")
    compare_arrays(float_int8, qdq_int8)

    # 6. 保存量化输出数据到 .bin
    os.makedirs(OUT_DIR, exist_ok=True)
    token_int8.tofile(f"{OUT_DIR}/input_int8.bin")
    float_int8.tofile(f"{OUT_DIR}/float_output_int8.bin")
    qdq_int8.tofile(f"{OUT_DIR}/qdq_output_int8.bin")
    float_out.astype(np.float16).tofile(f"{OUT_DIR}/float_output_f16.bin")
    qdq_out.astype(np.float16).tofile(f"{OUT_DIR}/qdq_output_f16.bin")
    print(f"\n量化输出已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
