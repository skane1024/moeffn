#!/usr/bin/env python3

import json

import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationMethod,
    QuantFormat,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
import os


class RandomCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, num_samples: int = 5000):
        self.input_name = input_name
        self.num_samples = num_samples
        self.current_idx = 0
        self.rng = np.random.default_rng(55)

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        data = self.rng.standard_normal(
            size=(1, 3072, 1, 72)
        ).astype(np.float32)
        self.current_idx += 1

        return {self.input_name: data}

    def rewind(self):
        self.current_idx = 0
        self.rng = np.random.default_rng(55)


def report_io_data_range(model_path: str, data_reader: CalibrationDataReader, max_samples: int = None):
    """
    用校准数据跑一遍原始模型, 统计整体输入/输出的数据范围 (min/max)。
    max_samples 为 None 时遍历全部校准样本。
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(model_path, so)
    input_name = session.get_inputs()[0].name

    data_reader.rewind()
    in_min, in_max = np.inf, -np.inf
    out_min, out_max = np.inf, -np.inf
    count = 0
    while True:
        if max_samples is not None and count >= max_samples:
            break
        feed = data_reader.get_next()
        if feed is None:
            break
        x = feed[input_name]
        y = session.run(None, feed)[0]
        in_min = min(in_min, float(x.min()))
        in_max = max(in_max, float(x.max()))
        out_min = min(out_min, float(y.min()))
        out_max = max(out_max, float(y.max()))
        count += 1

    # 统计会耗尽 reader, 复位后再交给量化校准使用
    data_reader.rewind()

    print(f"\n整体输入输出数据范围 (基于 {count} 个校准样本):")
    print(f"  输入 range: [{in_min:.6f}, {in_max:.6f}]")
    print(f"  输出 range: [{out_min:.6f}, {out_max:.6f}]")
    return {"input": (in_min, in_max), "output": (out_min, out_max)}


def export_quant_params(quantized_model_path: str, output_json_path: str = "quant_params.json"):
    """
    从量后的 ONNX 模型中提取所有量化参数。
    规则:
      - 激活 (QuantizeLinear 输入)  → int8
      - 权重 (DequantizeLinear 输入, *_quantized, 非 bias) → int8
      - bias  (DequantizeLinear 输入, *_bias_quantized)     → int32
      权重 scale 若为向量(shape>0) 则是 per-channel, 否则 per-tensor。
    输出格式:
        "tensor_name": {
            "dtype": "int8" | "int32",
            # per-tensor
            "scale": 0.043936651200056076,
            "zeropoint": -3,
            # 或 per-channel
            "scales": [0.01, 0.02, ...],
            "zeropoints": [0, 0, ...]
        }
    """
    model = onnx.load(quantized_model_path)
    init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

    def make_entry(tensor_input_name: str):
        """根据被量化 tensor 名判断 dtype 并提取 scale/zp。"""
        for node in model.graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear") and node.input[0] == tensor_input_name:
                scale_arr = init[node.input[1]]
                zp_name = node.input[2] if len(node.input) > 2 else None
                zp_arr = init[zp_name] if zp_name and zp_name in init else None

                # bias -> int32
                if "bias" in tensor_input_name:
                    dtype = "int32"
                else:
                    dtype = "int8"

                is_per_channel = scale_arr.ndim > 0 and scale_arr.size > 1

                if is_per_channel:
                    if zp_arr is not None and zp_arr.ndim > 0:
                        zps = zp_arr.tolist()
                    else:
                        zps = [int(zp_arr)] * scale_arr.size if zp_arr is not None else None
                    return {
                        "dtype": dtype,
                        "scales": scale_arr.tolist(),
                        "zeropoints": zps,
                    }
                else:
                    return {
                        "dtype": dtype,
                        "scale": float(scale_arr),
                        "zeropoint": int(zp_arr) if zp_arr is not None else 0,
                    }
        return None

    params = {}

    # 图输出名集合: 末端 Q/DQ 对的 DQ 输出即 graph output
    graph_output_names = {o.name for o in model.graph.output}
    # tensor 名 -> 消费它的节点
    consumer = {}
    for node in model.graph.node:
        for t in node.input:
            consumer.setdefault(t, []).append(node)

    def output_key(ql_node):
        """若 QuantizeLinear 的 Q->DQ 链末端是 graph output, 返回输出名, 否则 None.
        量化器会把 Conv 输出重命名为 xxx_QuantizeLinear_Input, 直接用该中间名不对,
        应以 graph output 的名字作为 key。"""
        for dq in consumer.get(ql_node.output[0], []):
            if dq.op_type == "DequantizeLinear":
                for out in dq.output:
                    if out in graph_output_names:
                        return out
        return None

    # 1) 激活: 遍历 QuantizeLinear 节点, 记录其输入 tensor
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            key = node.input[0]
            # graph 输出用输出名代替量化器生成的中间名
            out_name = output_key(node)
            if out_name is not None:
                key = out_name
            entry = make_entry(node.input[0])
            if entry is not None:
                params[key] = entry

    # 2) 权重/bias: 遍历 DequantizeLinear 节点, 只处理 *_quantized 权重/bias
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            key = node.input[0]
            if not key.endswith("_quantized"):
                continue  # 激活 DQ, 跳过
            # 还原原始权重/bias 名 (去掉 _quantized)
            orig_key = key[:-len("_quantized")]
            entry = make_entry(key)
            if entry is not None:
                params[orig_key] = entry


    with open(output_json_path, "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def main():
    model_path = "expert/expert_72.onnx"
    output_path = "expert/expert_72_qdq.onnx"

    model = onnx.load(model_path)
    input_name = model.graph.input[0].name

    data_reader = RandomCalibrationDataReader(
        input_name=input_name, num_samples=5000
    )

    # 量化前统计整体输入/输出数据范围
    report_io_data_range(model_path, data_reader)

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,  # Q/DQ 格式
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        per_channel=True,
        reduce_range=False,
    )

    export_quant_params(output_path, "expert/quant_params.json")

    original_size = os.path.getsize(model_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n量化完成!")
    print(f"  原始模型大小: {original_size:.2f} MB")
    print(f"  量化模型大小: {quantized_size:.2f} MB")
    print(f"  量化模型: {output_path}")
    print(f"  量化参数: quant_params.json")


if __name__ == "__main__":
    main()