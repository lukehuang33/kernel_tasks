"""Shared matmul benchmark logic."""

from __future__ import annotations

from contextlib import nullcontext

DEFAULT_DIM = 4096
DEFAULT_DTYPE = "bfloat16"
DTYPE_CHOICES = ("bfloat16",)
ACCUMULATION_DTYPE = "float32"
OUTPUT_DTYPE = "bfloat16"
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 100
USE_CUDA_GRAPH_REPLAY = True


def run_matmul_local(
    dim: int = DEFAULT_DIM,
    dtype: str = DEFAULT_DTYPE,
    profile_region=None,
) -> dict:
    """Execute a naive GEMM with BF16 inputs, FP32 accumulation, BF16 output."""
    import torch

    dim = int(dim)
    requested_dtype_name = str(dtype).lower()
    if requested_dtype_name not in (DEFAULT_DTYPE, "float16", "float32"):
        raise ValueError(
            f"Unsupported dtype '{requested_dtype_name}'. "
            f"kernel_1_naive only supports {DEFAULT_DTYPE} inputs."
        )
    dtype_name = DEFAULT_DTYPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    device = torch.device("cuda")
    input_dtype = torch.bfloat16
    output_dtype = torch.bfloat16

    a = torch.randn((dim, dim), device=device, dtype=input_dtype)
    b = torch.randn((dim, dim), device=device, dtype=input_dtype)
    c = torch.empty((dim, dim), device=device, dtype=output_dtype)
    torch_stream = torch.cuda.current_stream(device)

    # Warm up CUDA, cuBLAS, and GPU clocks before timing.
    for _ in range(WARMUP_ITERATIONS):
        torch.mm(a, b, out=c)
    torch_stream.synchronize()

    timed_matmul_region = profile_region or nullcontext
    profiler_enabled = profile_region is not None
    timed_iterations = 1 if profiler_enabled else TIMED_ITERATIONS
    use_cuda_graph_replay = USE_CUDA_GRAPH_REPLAY and not profiler_enabled
    cuda_graph = None

    if use_cuda_graph_replay:
        # Capture a warmed, allocation-free matmul so the timing reflects
        # device GEMM time rather than Python/dispatcher overhead.
        capture_stream = torch.cuda.Stream(device=device)
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            cuda_graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            torch.mm(a, b, out=c)
        capture_stream.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with timed_matmul_region():
        start_event.record(torch_stream)
        for _ in range(timed_iterations):
            if cuda_graph is not None:
                cuda_graph.replay()
            else:
                torch.mm(a, b, out=c)
        end_event.record(torch_stream)
        torch_stream.synchronize()

    total_elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_ms = total_elapsed_ms / timed_iterations
    elapsed_s = elapsed_ms / 1000.0

    # GEMM does 2 * M * N * K floating-point ops when counting mul+add separately.
    flops = 2 * dim * dim * dim
    tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0.0

    return {
        "status": "success",
        "operation": "torch.mm",
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "requested_dtype": requested_dtype_name,
        "dtype": dtype_name,
        "accumulation_dtype": ACCUMULATION_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "device": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "timing_method": (
            "cuda_events_graph_replay_average"
            if use_cuda_graph_replay
            else "cuda_events_explicit_stream_average"
        ),
        "profiler_enabled": profiler_enabled,
        "cuda_graph_replay": use_cuda_graph_replay,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": timed_iterations,
        "total_elapsed_ms": round(total_elapsed_ms, 3),
        "elapsed_ms": round(elapsed_ms, 3),
        "flops": flops,
        "tflops": round(tflops, 6),
        "result_checksum": float(c.float().sum().item()),
    }
