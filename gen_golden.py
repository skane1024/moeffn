import os
import numpy as np
from common import *
from moe_ffn_ref import expert_onnx_forward


IN_SCALE     = 0.047759633511304855   # 输入量化参数
IN_ZEROPOINT = 3


# prefill 阶段 golden: 1024 token, w72 专家模型, 只计算前 2 个专家
# use_expert=True  : 输入先送专家模型计算, 再加权求和
# use_expert=False : 跳过专家计算, 直接对输入数据加权求和
def gen_prefill_golden(out_dir="test_data", use_expert=True):
    NUM_TOKENS = 1024
    N_DIMS = 3072
    rng = np.random.default_rng(55)
    expert_path = "expert/expert_72_qdq.onnx" if use_expert else ""
    os.makedirs(out_dir, exist_ok=True)

    # token 数据
    token_data = rng.standard_normal(size=(NUM_TOKENS, N_DIMS)).astype(np.float16)
    token_data_np128 = to_np128(token_data, NUM_TOKENS, N_DIMS)
    token_data_np128.tofile(f"{out_dir}/token_data_np128_f16.bin")

    token_data_np128_int8, s, z = int8_quantize(token_data_np128, IN_SCALE, IN_ZEROPOINT)
    token_data_np128_int8.tofile(f"{out_dir}/token_data_np128_int8.bin")
    print("in scale_zeropoint:", IN_SCALE, IN_ZEROPOINT, np.min(token_data), np.max(token_data))

    # router 矩阵 (NUM_EXPERTS, NUM_TOKENS), 非零值 = 权重
    router_matrix = build_router_matrix(rng, NUM_TOKENS)
    # router_matrix[0][0] = 0.123
    # router_matrix[0][2] = 0.123
    # router_matrix[0][3] = 0.123
    # router_matrix[0][4] = 0.123
    # router_matrix[0][5] = 0.123
    # router_matrix[0][6] = 0.123
    # router_matrix[0][7] = 0.123
    # router_matrix[0][8] = 0.123

    router_matrix = router_matrix.astype(np.float16)
    router_matrix.tofile(f"{out_dir}/expert_tokens.bin")

    # golden 结果 (float32 累加, 与 reference 一致)
    golden_result = np.zeros((NUM_TOKENS, N_DIMS), dtype=np.float32)
    for expert_id in range(NUM_EXPERTS):
        token_ids, token_weights = scan_expert_tokens(router_matrix, expert_id)
        if len(token_ids) == 0:
            continue
        input_data = token_data[token_ids].astype(np.float32)
        golden_result[token_ids] += token_weights[:, None].astype(np.float32) * expert_onnx_forward(input_data, len(token_ids), N_DIMS, expert_path)

    golden_result_np128_f16 = to_np128(golden_result.astype(np.float16), NUM_TOKENS, N_DIMS)
    golden_result_np128_f16.tofile(f"{out_dir}/golden_result_np128_f16.bin")

    golden_result_np128_int8, out_scale, out_zeropoint = int8_quantize(golden_result_np128_f16)
    golden_result_np128_int8.tofile(f"{out_dir}/golden_result_np128_int8.bin")
    print("out scale_zeropoint:", out_scale, out_zeropoint)
    return out_scale, out_zeropoint



# generation 阶段 golden: 1 token, w1 专家模型, 计算全部专家, int8/router 数据补零
# use_expert=True  : 输入先送专家模型计算, 再加权求和
# use_expert=False : 跳过专家计算, 直接对输入数据加权求和
def gen_generation_golden(out_dir="test_data_generation", use_expert=True):
    NUM_TOKENS = 1
    N_DIMS = 3072
    rng = np.random.default_rng(42)
    expert_path = "expert/expert_1_qdq.onnx" if use_expert else ""
    os.makedirs(out_dir, exist_ok=True)
    # 生成随机的输入数据 token 数据
    token_data = rng.standard_normal(size=(NUM_TOKENS, N_DIMS)).astype(np.float16)
    token_data_np128 = to_np128(token_data, NUM_TOKENS, N_DIMS)
    token_data_np128.tofile(f"{out_dir}/token_data_np128_f16.bin")

    token_data_np128_int8, s, z = int8_quantize(token_data_np128, IN_SCALE, IN_ZEROPOINT)
    token_data_np128_int8.tofile(f"{out_dir}/token_data_np128_int8.bin")
    print("in scale_zeropoint:", IN_SCALE, IN_ZEROPOINT, np.min(token_data), np.max(token_data))

    # router 矩阵 (NUM_EXPERTS, NUM_TOKENS), 非零值 = 权重
    router_matrix = build_router_matrix(rng, NUM_TOKENS)
    router_matrix.tofile(f"{out_dir}/expert_tokens.bin")

    # golden 结果
    golden_result = np.zeros((NUM_TOKENS, N_DIMS), dtype=np.float32)
    for expert_id in range(NUM_EXPERTS):
        token_ids, token_weights = scan_expert_tokens(router_matrix, expert_id)
        if len(token_ids) == 0:
            continue
        print(expert_id, token_weights)
        input_data = token_data[token_ids].astype(np.float32)
        golden_result[token_ids] += token_weights[:, None].astype(np.float32) * expert_onnx_forward(input_data, len(token_ids), N_DIMS, expert_path)
        print(np.min(golden_result), np.max(golden_result))

    golden_result_np128 = to_np128(golden_result.astype(np.float16), NUM_TOKENS, N_DIMS)
    golden_result_np128.tofile(f"{out_dir}/golden_result_np128_f16.bin")

    golden_result_np128_int8, out_scale, out_zeropoint = int8_quantize(golden_result_np128)
    golden_result_np128_int8.tofile(f"{out_dir}/golden_result_np128_int8.bin")
    print("out scale_zeropoint:", out_scale, out_zeropoint)


if __name__ == "__main__":
    gen_prefill_golden()
    gen_generation_golden()
