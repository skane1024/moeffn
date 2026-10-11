import onnx

m = onnx.load("expert/expert_72_qdq.onnx")

# 直接修改 input/output 声明的 shape: 最后一维 72 -> 1
for io in list(m.graph.input) + list(m.graph.output):
    dims = io.type.tensor_type.shape.dim
    if dims[-1].HasField("dim_value") and dims[-1].dim_value == 72:
        dims[-1].dim_value = 1

# 重新推理中间节点的 shape, 校验后保存
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, "expert/expert_1_qdq.onnx")