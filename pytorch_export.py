import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import numpy_helper


# 保存新模型的目录
SAVE_DIR = "expert"

# ONNX 图输入 / 输出名
INPUT_NAME = "input"
OUTPUT_NAME = "output"

# 中间张量名（与原模型一致，便于量化工具链 / 调试对齐）
OUT_CONV1 = "conv_1_out"
OUT_SLICE0 = "5"
OUT_SLICE1 = "8"
OUT_SIGMOID = "6"
OUT_MUL0 = "7"
OUT_MUL1 = "9"

# Conv 权重 initializer 名（与原模型一致；量化工具会基于该名字生成 *_quantized）
WEIGHT_NAME_CONV1 = "conv1.weight"
WEIGHT_NAME_CONV3 = "conv3.weight"

# 随机权重取值范围（仅 --random 模式使用）
WEIGHT_MIN = -0.04
WEIGHT_MAX = 0.03

# 随机种子（保证可复现）
SEED = 55

# 导出 opset（主域与原模型一致）
OPSET = 13


class SwiGLUFFN(nn.Module):
    """与原 model_1core_opt_w72_new.onnx 结构一致的 SwiGLU Gated FFN。

    结构:
      conv1 :  (B, 3072, 1, 72) -> (B, 640, 1, 72)          # gate+up 拼接
      split :  切成 (B, 320, 1, 72) gate, (B, 320, 1, 72) up
      silu  :  gate * sigmoid(gate)                          # = SiLU(gate)
      mul   :  silu(gate) * up                               # SwiGLU 门控
      conv3 :  (B, 320, 1, 72) -> (B, 3072, 1, 72)           # down projection
    """

    def __init__(self):
        super().__init__()
        # 无 bias，与原模型 Conv 节点一致（只接受 data + W 两个输入）
        self.conv1 = nn.Conv2d(3072, 640, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(320, 3072, kernel_size=1, bias=False)

    def forward(self, x):
        h = self.conv1(x)                        # (B, 640, 1, 72)
        gate = h[:, :320, :, :]                  # (B, 320, 1, 72)
        up = h[:, 320:, :, :]                    # (B, 320, 1, 72)
        gate_out = gate * torch.sigmoid(gate)    # SiLU(gate)
        hidden = gate_out * up                   # SwiGLU 门控
        return self.conv3(hidden)                # (B, 3072, 1, 72)


def load_weights_from_onnx(model, onnx_path):
    """从已有 ONNX 模型读取 conv1 / conv3 权重，载入 PyTorch 模型，保证数值一致。"""
    m = onnx.load(onnx_path)
    weights = {}
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if "conv1.weight" in init.name:
            weights["conv1"] = arr
        elif "conv3.weight" in init.name:
            weights["conv3"] = arr
    assert set(weights) == {"conv1", "conv3"}, f"未找到完整权重, 得到 {list(weights)}"
    for init_name, arr in weights.items():
        param = getattr(model, {"conv1": "conv1", "conv3": "conv3"}[init_name]).weight
        with torch.no_grad():
            param.copy_(torch.from_numpy(arr))   # ONNX [out,in,1,1] 与 PyTorch 布局一致


def init_random_weights(model, weight_min, weight_max, seed):
    """按均匀分布重设两个 Conv 权重。"""
    rng = np.random.default_rng(seed)
    for m in (model.conv1, model.conv3):
        w = m.weight
        new_arr = rng.uniform(weight_min, weight_max, size=tuple(w.shape)).astype(np.float32)
        with torch.no_grad():
            w.copy_(torch.from_numpy(new_arr))


def rename_inputs_outputs(model, name_map):
    """把图输入/输出张量重命名，并同步更新所有节点的引用。"""
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in name_map:
                node.input[i] = name_map[name]
        for i, name in enumerate(node.output):
            if name in name_map:
                node.output[i] = name_map[name]
    for gio in list(model.graph.input) + list(model.graph.output):
        if gio.name in name_map:
            gio.name = name_map[gio.name]
    return model


def rename_tensors_to_original(model):
    """把导出图的中张量名 / 权重名对齐到原模型。

    量化工具链会用权重 initializer 名生成对应的 *_quantized / *_scale /
    *_zero_point。若权重名叫 'conv1.weight'，量化器生成的 DequantizeLinear
    输入类型 float 而非法；必须使用与原模型一致的权重名，中间张量名同步对齐。
    """
    tensor_map = {
        "/conv1/Conv_output_0": OUT_CONV1,
        "/Slice_output_0": OUT_SLICE0,
        "/Slice_1_output_0": OUT_SLICE1,
        "/Sigmoid_output_0": OUT_SIGMOID,
        "/Mul_output_0": OUT_MUL0,
        "/Mul_1_output_0": OUT_MUL1,
        "conv1.weight": WEIGHT_NAME_CONV1,
        "conv3.weight": WEIGHT_NAME_CONV3,
    }
    # 图内张量重命名（含 Conv 节点上下文），refine_graph_name 逐个应用
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in tensor_map:
                node.input[i] = tensor_map[name]
        for i, name in enumerate(node.output):
            if name in tensor_map:
                node.output[i] = tensor_map[name]
    for gio in list(model.graph.input) + list(model.graph.output):
        if gio.name in tensor_map:
            gio.name = tensor_map[gio.name]
    for init in model.graph.initializer:
        if init.name in tensor_map:
            init.name = tensor_map[init.name]
    return model


def fold_slice_constants(model):
    """把 PyTorch 导出的 Constant+Unsqueeze 链折叠成 initializer。

    torchscript exporter 会把 slice 的 starts/ends/axes/steps 拆成
    「标量 Constant -> Unsqueeze(axes=[0])」的节点链；原模型（torch-jit
    优化导出）是直接把这些边界折叠成 int64 initializer，且 Slice 只给
    4 个输入（省略全 1 的 steps）。本函数把导出的图还原成原模型的结构。
    """
    g = model.graph

    # 1) 收集 Constant 值
    const_values = {}
    for node in g.node:
        if node.op_type == "Constant":
            const_values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    # 2) 收集 Unsqueeze(constant) 的展开结果：输出名 -> 1D numpy 数组
    folded = {}
    for node in g.node:
        if node.op_type == "Unsqueeze" and node.input[0] in const_values:
            axes = None
            if len(node.input) > 1 and node.input[1] in const_values:
                axes = np.atleast_1d(const_values[node.input[1]])
            elif len(node.attribute) > 0:
                for a in node.attribute:
                    if a.name == "axes":
                        axes = numpy_helper.to_array(a.t)
            if axes is not None:
                folded[node.output[0]] = np.expand_dims(
                    const_values[node.input[0]], tuple(int(x) for x in axes))

    # 3) 重写 Slice：常量边界 -> 去重后的 initializer；去掉全 1 的 steps
    value_to_init = {}

    def get_init(arr):
        arr = arr.astype(arr.dtype, copy=False)
        key = (arr.dtype, arr.shape, arr.tobytes())
        if key not in value_to_init:
            name = f"_v_{len(g.initializer)}"
            g.initializer.append(numpy_helper.from_array(arr, name=name))
            value_to_init[key] = name
        return value_to_init[key]

    for node in g.node:
        if node.op_type != "Slice":
            continue
        raw_inputs = list(node.input)
        # 逐个解析边界输入（第 1~4 个输入）：常量 -> 值，动态 -> 保留原名
        vals = []
        for inp in raw_inputs[1:]:
            if inp in folded:
                vals.append(("val", folded[inp]))
            elif inp in const_values:
                vals.append(("val", const_values[inp]))
            else:
                vals.append(("name", inp))
        # 若最后一个输入（steps）是全 1，删掉 —— 与原模型 4 输入一致
        if len(vals) == 4 and vals[3][0] == "val" and np.all(vals[3][1] == 1):
            vals = vals[:3]
        # 重建 Slice 输入
        new_inputs = [raw_inputs[0]]
        for kind, v in vals:
            new_inputs.append(get_init(v) if kind == "val" else v)
        del node.input[:]
        node.input.extend(new_inputs)

    # 4) 删除不再被引用的 Constant / Unsqueeze 节点（迭代直到稳定）
    changed = True
    while changed:
        changed = False
        used = {inp for n in g.node for inp in n.input if inp}
        surviving = []
        for n in g.node:
            if n.op_type in ("Constant", "Unsqueeze") and not any(
                    out in used for out in n.output):
                changed = True
                continue
            surviving.append(n)
        if changed:
            del g.node[:]
            g.node.extend(surviving)

    return model


def export_onnx(model, save_path, input_shape=(1, 3072, 1, 72)):
    """导出 ONNX，图输入输出名与原模型对齐。

    PyTorch 的 ONNX 导出不允许数字开头 / 非标识符的名字，
    因此先用合法名字导出，再把输入输出重命名回 '4' / '10'。
    导出后折叠 Constant/Unsqueeze 链，使 Slice 结构与原模型一致。
    """
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    tmp_path = save_path + ".tmp"
    # 旧版 (torchscript) 导出器 + keep_initializers_as_inputs=False：
    # 权重 / slice 边界等常量全部保留为 graph.initializer，
    # 图输入只有 '4'，与原模型一致（不引入额外 imports）。
    torch.onnx.export(
        model,
        dummy,
        tmp_path,
        input_names=["input_4"],
        output_names=["output_10"],
        opset_version=OPSET,
        do_constant_folding=False,          # 保持原始算子结构
        keep_initializers_as_inputs=False,  # 常量进 initializer，不当作图输入
    )

    # 重命名为原模型的名字，并折叠 slice 常量链
    m = onnx.load(tmp_path)
    m = fold_slice_constants(m)
    m = rename_inputs_outputs(m, {"input_4": INPUT_NAME, "output_10": OUTPUT_NAME})
    m = rename_tensors_to_original(m)
    os.remove(tmp_path)

    onnx.checker.check_model(m)
    onnx.save(m, save_path)
    print(f"[导出完成] {save_path}")


def print_graph(model_path):
    """打印 ONNX 图结构。"""
    m = onnx.load(model_path)
    g = m.graph
    print(f"== {os.path.basename(model_path)} 图结构 ==")
    for i in g.input:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"  input {i.name} {shape}")
    for n in g.node:
        print(f"  {n.op_type:8s} in={list(n.input)}  out={list(n.output)}")
    for o in g.output:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  output {o.name} {shape}")


def verify_pytorch_vs_onnx(pt_model, onnx_path):
    """同一随机输入下，对比 PyTorch forward 与导出的 ONNX 推理结果。"""
    pt_model.eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3072, 1, 72)).astype(np.float32)
    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x)).numpy()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so)
    y_onnx = sess.run(None, {INPUT_NAME: x})[0]

    mae = float(np.mean(np.abs(y_pt - y_onnx)))
    max_err = float(np.max(np.abs(y_pt - y_onnx)))
    print(f"[数值对比] PyTorch vs ONNX  MAE={mae:.3e}  max_err={max_err:.3e}")
    status = "一致 ✓" if max_err < 1e-5 else "不一致 ✗"
    print(f"  结果: {status}")


if __name__ == "__main__":
    model = SwiGLUFFN()

    # 默认从参考 clean 模型加载权重（保证与现有 ONNX 完全一致）；
    # 加 --random 则用均匀分布 [-0.04, 0.03] 重新生成。
    init_random_weights(model, WEIGHT_MIN, WEIGHT_MAX, SEED)
    print("使用均匀分布权重初始化 (rng.uniform(-0.04, 0.03))")


    save_path = os.path.join(SAVE_DIR, "model_1core_opt_w72_new_pytorch.onnx")
    export_onnx(model, save_path)
    print_graph(save_path)
    verify_pytorch_vs_onnx(model, save_path)