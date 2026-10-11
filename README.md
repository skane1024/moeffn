MOE-FFN的整个测试过程

1. 生成专家模型，每个专家生成随机的onnx模型 pytorch_export.py, 前期有一个就够了，只是权重不一样，后续可以根据需要生成更多的专家模型。
2. 量化专家模型，生成量化后的onnx模型 onnx_quant.py 得到专家的量化参数 quant_params.json
3. 生成测试数据，生成随机的输入数据，计算输出结果，gen_golden.py验证模型的正确性。 得到MOE-FFN的输入输出量化参数
4. 根据MOE-FFN的输入输出量化参数，以及专家的模型和量化参数，送入编译器编译MBS模型
5. 根据MBS模型，使用python脚本做一个拼接(moe-ffn和专家是不同的mbs),得到一个完整的mbs,拼接的时候有很多注意的点
   - 如果没有权重buffer，需要在拼接的时候手动添加权重buffer  权重的地址需要4K对齐
   - 如果没有temp buffer，需要在拼接的时候手动添加temp buffer  temp buffer的地址需要4K对齐
   - 需要检查专家的reg_map是否和算子实现一致
   - CMD拼接的时候， 每一个CMD的起始地址需要4K对齐
6. 需要检查moe-ffn算子的参数是都正确



遇到的坑

1. 回发现NPU计算出来的结果误差很大，这个是生成的onnx模型导致的，需要使用QDQ模型去生成golden数据
