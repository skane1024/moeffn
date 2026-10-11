import numpy as np

NUM_TOKENS   = 1             # token 总数
N_DIMS       = 3072             # 每 token 特征维度
NUM_EXPERTS  = 116              #专家的个数
TOP_K        = 8                # 每个 token 被路由到的专家数

rng = np.random.default_rng(55)

expert_path = "expert/model_1core_opt_w1_int8.onnx"
# expert_path = ""


import numpy as np

def int8_quantize(data,scale = 0.0, zero_point = 0.0):
    x = np.asarray(data, dtype=np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    
    if scale == 0.0:
        scale = max(np.abs(x_min), np.abs(x_max)) / 127.0
        zero_point = 0
    q = np.round(x / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    if scale == 0.0:
        return q, scale, zero_point
    return q, scale, zero_point

def to_np128(data):
    data = data.reshape((NUM_TOKENS, N_DIMS))
    data = data.transpose((1,0))
    data = data.reshape(N_DIMS // 128,128, NUM_TOKENS)
    data = data.transpose(0,2,1)
    return data




# 从 router 矩阵扫描当前专家的 (token_id, weight)
def scan_expert_tokens(router_matrix, expert_id):
    expert_row = router_matrix[expert_id]
    token_ids  = np.nonzero(expert_row)[0]
    weights    = expert_row[token_ids]
    return token_ids, weights


# 专家计算
def expert_forward(input_data, expert_onnx_model=""):
    print(input_data.shape)
    if expert_onnx_model != "":
        import onnxruntime as ort
        input_data_1 = input_data.transpose((1,0))
        input_data_1 = input_data_1.reshape((1,3072,1,1))
        input_data_1 = input_data_1.astype(np.float32)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(expert_onnx_model, so)
        input_name_0 = session.get_inputs()[0].name
        input_data_map = {input_name_0: input_data_1 }
        outputs = session.run(None, input_data_map)[0]
        outputs = outputs.transpose(0,3,2,1)
        outputs = outputs.reshape(1,3072)
        print("expert result: ", np.min(outputs), np.max(outputs))
        return outputs

    return  input_data


#token 数据
token_data = rng.standard_normal(size=(NUM_TOKENS, N_DIMS)).astype(np.float16)
token_data_np128 = to_np128(token_data)
token_data_np128.tofile("test_data/token_data_np128_f16.bin")

# token_data_np128_int8,in_scale,in_zeropoint = int8_quantize(token_data_np128)
in_scale = 0.047759633511304855
in_zeropoint = 3
token_data_np128_int8,s,z = int8_quantize(token_data_np128, in_scale, in_zeropoint)
token_data_np128_int8.tofile("test_data/token_data_np128_int8.bin")
print("in scale_zeropoint:", in_scale,in_zeropoint, np.min(token_data), np.max(token_data))


# 2. router 矩阵 (NUM_EXPERTS, NUM_TOKENS), 非零值 = 权重
router_matrix = np.fromfile("test_data/expert_tokens.bin", dtype = np.float16).reshape((NUM_EXPERTS, NUM_TOKENS))


NPU_input = []

golden_result = np.zeros((NUM_TOKENS, N_DIMS), dtype=np.float32)
for expert_id in range(2):
    token_ids, token_weights = scan_expert_tokens(router_matrix, expert_id)
    print(token_ids)
    print(len(token_ids))
    input_data = token_data[token_ids].astype(np.float32)
    input_data_np128_int8 = token_data_np128_int8[:,token_ids,:]
    input_data_np128_int8.tofile(f"test_data/npu_input_{expert_id}.bin")
    npe_result =expert_forward(input_data,expert_path)
    npe_result_np128 = npe_result
    npe_result_np128 = npe_result_np128.reshape((72, N_DIMS))
    npe_result_np128 = npe_result_np128.transpose((1,0))
    npe_result_np128 = npe_result_np128.reshape(N_DIMS // 128,128, 72)
    npe_result_np128 = npe_result_np128.transpose(0,2,1)
    npe_result_np128,s,z = int8_quantize(npe_result_np128, scale = 0.032091207802295685, zero_point = 60)
    npe_result_np128.tofile(f"test_data/npu_out_{expert_id}.bin")
    golden_result[token_ids] += token_weights[:, None].astype(np.float32) * npe_result


golden_result = golden_result.astype(np.float16)
golden_result_np128 = to_np128(golden_result)
golden_result_np128.tofile("test_data/golden_result_np128_f16.bin")


golden_result_np128_int8,out_scale,out_zeropoint = int8_quantize(golden_result_np128)
# golden_result_np128_int8 = int8_quantize(golden_result_np128, out_scale, out_zeropoint)
golden_result_np128_int8.tofile("test_data/golden_result_np128_int8.bin")
print("out scale_zeropoint:", out_scale,out_zeropoint)




data = np.fromfile("test_data/out_img_cast1_r1_i0.out", dtype  = np.int8)

phone_result = data[:3072 * 1024]
phone_npu_io = data[3072 * 1024:3072 * 1024 + 3072 * 72 *4]
phone_npu_io = phone_npu_io.reshape(4,3072 * 72)
phone_npu_in_1 = phone_npu_io[0, :]
phone_npu_in_2 = phone_npu_io[2, :]
phone_npu_in_1.tofile("test_data/phone_npu_in_1.bin")
phone_npu_in_2.tofile("test_data/phone_npu_in_2.bin")

phone_npu_out_1 = phone_npu_io[1, :]
phone_npu_out_2 = phone_npu_io[3, :]
phone_npu_out_1.tofile("test_data/phone_npu_out_1.bin")
phone_npu_out_2.tofile("test_data/phone_npu_out_2.bin")

print(phone_npu_out_1)
print(phone_npu_out_2)