import numpy as np
import onnxruntime as ort


rng = np.random.default_rng(55)


def int8_quantize11(data, scale, zero_point):
    x = np.asarray(data, dtype=np.float32)

    q = np.round(x / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q

input = rng.standard_normal(size=(1,3072,1,1)).astype(np.float32)

in_scale = 0.047759633511304855
in_zeropoint = 3
token_data = int8_quantize11(input,in_scale, in_zeropoint)
token_data.tofile("input_1.bin")


token_data_np128 = token_data.reshape(1,24,128,1)
token_data_np128 = token_data_np128.transpose((0,1,3,2))
token_data_np128 = np.pad(token_data_np128, ((0,0),(0,0),(0,1),(0,0)), mode="constant",constant_values=0)
token_data_np128.tofile("input_1_np128.bin")

so = ort.SessionOptions()
so.log_severity_level = 3
session = ort.InferenceSession("model_1core_opt_w1.onnx", so)
input_name_0 = session.get_inputs()[0].name
input_data_map = {input_name_0: input }
outputs = session.run(None, input_data_map)[0]


out_scale = 0.032091207802295685
out_zeropoint = 60
outputs =int8_quantize11(outputs, out_scale, out_zeropoint)
outputs.tofile("output_1.bin")

outputs_np128 = outputs.reshape(1,24,128,1)
outputs_np128 = outputs_np128.transpose((0,1,3,2))
outputs_np128 = np.pad(outputs_np128, ((0,0),(0,0),(0,1),(0,0)), mode="constant",constant_values=0)
outputs_np128.tofile("output_1_np128.bin")