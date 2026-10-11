在这个项目中，我需要构建MOE-FFN的计算代码，生成随机的输入数据，计算输出结果，验证模型的正确性。 gen_golden_generation.py和gen_golden.py已经有了一些代码可以参考，
  每个专家生成随机的onnx模型 pytorch_export.py代码中已经有了相关的代码，只需要根据需要修改即可。


NP128 是NPU输入数据的一个特殊排布，每个物理块有128维特征，对于3074个维度的数据，每个token有24个物理块。
NUM_TOKENS   = 1024          # token 总数
N_DIMS          = 3072         # 每 token 特征维度
PACK         = 128          # NP128: 每物理块 128 维

NUM_EXPERTS  = 116
每个token选择8个专家




