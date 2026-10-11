import numpy as np


data = np.fromfile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/out_img_cast1_r1_i0.out",dtype = np.int8)

data_input = np.fromfile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/token_data_np128_int8.bin",dtype = np.int8)

data_result = data[:3072*2]
print(data_result)
data_npu_input = data[3072*2: 3072*4]
data_input = data_input[:3072*2]
print(np.max(np.abs(data_npu_input - data_input)))
data_input.tofile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/npu_input.bin")

data_npu_result1 = data[3072*4:3072*6]
print(data_npu_result1)
data_npu_result1.tofile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/npu_result1.bin")

data_npu_result2 = data[3072*6:3072*8]
print(data_npu_result2)
data_npu_result2.tofile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/npu_result2.bin")
print(np.max(np.abs(data_npu_result1 - data_npu_result2)))



# data_npu_result2 = data[3072*8:3072*10]
# print(data_npu_result2)
# data_npu_result2.tofile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/npu_result4.bin")
# print(np.max(np.abs(data_npu_result1 - data_npu_result2)))

# data_npu_result2 = data[3072*10:3072*12]
# print(data_npu_result2)
# data_npu_result2.tofile("/home/yongkang.shu/NPU_IN_VDSP/mbs_parser/parse_case/moe/moe1/npu_result5.bin")
# print(np.max(np.abs(data_npu_result1 - data_npu_result2)))
