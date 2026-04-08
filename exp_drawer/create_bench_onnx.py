import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

CALIB_DIR   = "calib_data"
CALIB_COUNT = 100

# ---------------------------------------------------------------------------
# compute_bench.onnx
# 8 stacked 1x1 Convs  [1, 1024, 64, 64] → ... → [1, 1024, 64, 64]
# FLOPs  = 8 × 2 × 1024 × 1024 × 64 × 64  ≈  34.4 GOPS
# Weight (int8 per layer) = 1024×1024 = 1MB → fits in NPU SPM → high reuse
# AI ≈ 8192 ops/byte >> ridge (~1200) → compute-bound
# @ 80 TOPS theoretical minimum latency: ~0.43 ms  (vs 3.64 ms for old 2.15 GOPS model)
# ---------------------------------------------------------------------------
def create_compute_bench():
    C, H, W, N_LAYERS = 1024, 64, 64, 8

    nodes, initializers = [], []
    prev = "input"
    for i in range(N_LAYERS):
        w_name   = f"weight_{i}"
        b_name   = f"bias_{i}"
        out_name = "output" if i == N_LAYERS - 1 else f"feat_{i}"

        w = (np.random.randn(C, C, 1, 1) * 0.01).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)
        initializers.append(numpy_helper.from_array(w, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))

        nodes.append(helper.make_node(
            "Conv",
            [prev, w_name, b_name],
            [out_name],
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
            name=f"conv_{i}"
        ))
        prev = out_name

    x   = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, C, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])

    graph = helper.make_graph(nodes, "compute_bench", [x], [out],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, "compute_bench.onnx")
    print("compute_bench.onnx created (IR v10, Opset 11)")
    print(f"  {N_LAYERS}x 1x1 Conv  [1,{C},{H},{W}]")
    print(f"  FLOPs: {N_LAYERS*2*C*C*H*W/1e9:.2f} GOPS  |  AI: ~{2*H*W:.0f} ops/byte")

    calib_path = os.path.join(CALIB_DIR, "compute_bench")
    os.makedirs(calib_path, exist_ok=True)
    for i in range(CALIB_COUNT):
        arr = np.random.uniform(-1, 1, (1, C, H, W)).astype(np.float32)
        arr_nhwc = arr.transpose(0, 2, 3, 1)[0]   # [H, W, C]
        np.save(os.path.join(calib_path, f"sample_{i:04d}.npy"), arr_nhwc)


# ---------------------------------------------------------------------------
# bandwidth_bench.onnx
# Depthwise Conv  [1, 256, 512, 512],  weight [256, 1, 3, 3]
# FLOPs  = 2 × 256 × 512 × 512 × 9  ≈  1.2 GOPS
# Activations (int8) = 256 × 512 × 512 × 2 ≈  128 MB >> any NPU SPM
# AI ≈ 9 ops/byte << ridge (~1200) → bandwidth-bound
# @ 66.7 GB/s theoretical minimum latency: ~1.9 ms  (>>overhead)
# ---------------------------------------------------------------------------
def create_bandwidth_bench():
    C, H, W = 256, 512, 512
    kH, kW  = 3, 3

    weight_np = (np.random.randn(C, 1, kH, kW) * 0.01).astype(np.float32)
    bias_np   = np.zeros(C, dtype=np.float32)

    weight = numpy_helper.from_array(weight_np, name="weight")
    bias   = numpy_helper.from_array(bias_np,   name="bias")

    x   = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, C, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])

    node = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        kernel_shape=[kH, kW],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=C
    )

    graph = helper.make_graph([node], "bandwidth_bench", [x], [out],
                              initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, "bandwidth_bench.onnx")
    print("bandwidth_bench.onnx created (IR v10, Opset 11)")

    flops = 2 * C * H * W * kH * kW
    act_b = C * H * W * 2
    print(f"  Depthwise Conv  [1,{C},{H},{W}]")
    print(f"  FLOPs: {flops/1e9:.2f} GOPS  |  Activations: {act_b/1e6:.0f} MB  |  AI: ~{flops/act_b:.1f} ops/byte")

    calib_path = os.path.join(CALIB_DIR, "bandwidth_bench")
    os.makedirs(calib_path, exist_ok=True)
    for i in range(CALIB_COUNT):
        arr = np.random.uniform(-1, 1, (1, C, H, W)).astype(np.float32)
        arr_nhwc = arr.transpose(0, 2, 3, 1)[0]   # [H, W, C]
        np.save(os.path.join(calib_path, f"sample_{i:04d}.npy"), arr_nhwc)


if __name__ == "__main__":
    np.random.seed(42)
    create_compute_bench()
    create_bandwidth_bench()
    print("\nDone. Now try compiling with compile_bench.py")
