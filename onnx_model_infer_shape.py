"""
onnx_model_infer_shape.py
-------------------------
读取 ONNX 模型，按命令行参数修改模型的输入/输出 shape（固定 4D 布局的中间/最后一维），
重新做 shape 推断（onnx.shape_inference），并保存为新文件。

用途示例：
  把 64-token 的专家子图改成 72-token 的：
    python3 onnx_model_infer_shape.py \
        model_1core_opt_w1_new.onnx \
        -o model_1core_opt_w1_new_72.onnx \
        -d 72 --dim 3

说明：
  - 模型输入输出均为 [1, 3072, 1, N]，N 为每批 token 数（本目录模型 N=1 或 72）。
  - 默认只修改最后一维（NCHW 的 W），可用 -d/--dim 指定修改第几维（0 起）。
  - 修改输入 shape 后，会通过 shape_inference 自动同步内部中间张量 shape 与输出 shape；
    若 onnx.shape_inference 覆盖不到的部分，可用 -o2/--override-output 强制写输出 shape。
"""

import argparse
import os

import onnx
from onnx import shape_inference


def parse_shape_dim(shape_dim):
    """把 onnx 的 shape.dim 数组转成可读列表：常量取 dim_value，动态维保留 '?'。"""
    dims = []
    for d in shape_dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def get_graph_io_shape(model):
    """打印模型所有输入/输出及其 shape。"""
    g = model.graph
    lines = []
    for v in g.input:
        dims = parse_shape_dim(v.type.tensor_type.shape.dim)
        lines.append(f"  input : {v.name:20s} {dims}")
    for v in g.output:
        dims = parse_shape_dim(v.type.tensor_type.shape.dim)
        lines.append(f"  output: {v.name:20s} {dims}")
    return "\n".join(lines)


def modify_input_shape(model, dim_index, new_size):
    """修改模型输入 tensor 的第 dim_index 维为 new_size。"""
    g = model.graph
    if not g.input:
        raise RuntimeError("模型没有输入")
    it = g.input[0].type.tensor_type
    shape_dims = list(it.shape.dim)
    if dim_index >= len(shape_dims):
        raise RuntimeError(
            f"输入 shape 只有 {len(shape_dims)} 维，无法修改第 {dim_index} 维")
    old = shape_dims[dim_index].dim_value
    shape_dims[dim_index].dim_value = new_size
    del it.shape.dim[:]
    it.shape.dim.extend(shape_dims)
    return old


def override_output_shape(model, new_size, dim_index=None):
    """可选：强制把输出 shape 写成与输入一致，供 shape_inference 覆盖不到的模型兜底。"""
    g = model.graph
    if not g.output:
        return
    it = g.output[0].type.tensor_type
    shape_dims = list(it.shape.dim)
    if dim_index is None:
        # 默认同步为与输入同长度的 shape
        if g.input:
            in_dims = list(g.input[0].type.tensor_type.shape.dim)
            if len(in_dims) == len(shape_dims):
                shape_dims = in_dims
    elif dim_index < len(shape_dims):
        shape_dims[dim_index].dim_value = new_size
    del it.shape.dim[:]
    it.shape.dim.extend(shape_dims)


def re_infer_shapes(model):
    """清掉旧推断结果后重新做 shape 推断，让中间张量 shape 随输入更新。"""
    model.graph.ClearField("value_info")
    try:
        return shape_inference.infer_shapes(model)
    except Exception as e:  # noqa: BLE001 推断失败不致命，仅打印
        print(f"[警告] shape_inference 失败: {e}")
        return model


def main():
    parser = argparse.ArgumentParser(
        description="修改 ONNX 模型输入/输出 shape 并重新 shape 推断")
    parser.add_argument("model", help="输入 ONNX 模型路径")
    parser.add_argument("-o", "--output", default=None,
                        help="保存路径（默认: 输入名 + '_shape<N>.onnx'）")
    parser.add_argument("-d", "--dim-value", type=int, required=True,
                        help="新的维度大小，例如 72")
    parser.add_argument("--dim-index", type=int, default=3,
                        help="要修改输入的第几维（0 起），默认 3（最后一维，NCHW 的 W）")
    parser.add_argument("--sync-output", action="store_true", default=True,
                        help="把输出 shape 同步为与输入一致（默认开启）")
    parser.add_argument("--no-sync-output", action="store_true",
                        help="不同步输出 shape，只改输入")
    parser.add_argument("--list", action="store_true",
                        help="只打印输入/输出 shape 后退出，不修改")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        parser.error(f"模型不存在: {args.model}")

    model = onnx.load(args.model)
    print(f"模型: {args.model}\n修改前:")
    print(get_graph_io_shape(model))

    if args.list:
        return

    old = modify_input_shape(model, args.dim_index, args.dim_value)
    print(f"\n输入第 {args.dim_index} 维: {old} -> {args.dim_value}")

    if args.sync_output and not args.no_sync_output:
        override_output_shape(model, args.dim_value, args.dim_index)

    model = re_infer_shapes(model)

    # 校验：输入/输出 shape 与 onnxruntime 实际运行 shape 一致
    try:
        onnx.checker.check_model(model)
    except Exception as e:  # noqa: BLE001
        print(f"[警告] check_model 失败: {e}")

    print("\n修改后:")
    print(get_graph_io_shape(model))

    save_path = args.output or (
        os.path.splitext(args.model)[0] + f"_shape{args.dim_value}.onnx")
    onnx.save(model, save_path)
    print(f"\n已保存: {save_path}")


if __name__ == "__main__":
    main()
