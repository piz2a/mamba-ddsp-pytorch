# DDSP ONNX Runtime C++ Benchmark

This experiment exports the **stateful one-frame neural controller** of each
model to ONNX and measures it in ONNX Runtime C++ with one CPU thread.

It deliberately excludes audio-rate synthesis. The current PyTorch synthesis
uses complex FFT operators (`view_as_complex`, `rfft`, and `irfft`) that do not
export through the installed PyTorch ONNX exporter. This is the same deployment
boundary used conceptually by `ddsp-vst`: neural inference produces synthesis
parameters, then native C++ DSP renders audio.

The exported GRU hidden state is an explicit input and output, so each graph can
preserve causal state in a real-time host. The benchmark currently times
`Session::Run`, including output allocation, on CPUExecutionProvider.

```bash
python ddsp_onnx_benchmark/export_controllers.py
cmake -S ddsp_onnx_benchmark -B ddsp_onnx_benchmark/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.18.1
cmake --build ddsp_onnx_benchmark/build -j
ddsp_onnx_benchmark/build/ddsp_onnx_benchmark \
  ddsp_onnx_benchmark/models runs/ddsp_realtime_latency_onnx_cpp
```
