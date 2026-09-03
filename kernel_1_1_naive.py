from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import sys
import types

DEFAULT_DIM = 4096
DEFAULT_DTYPE = "bfloat16"
DTYPE_CHOICES = ("bfloat16",)
ACCUMULATION_DTYPE = "float32"
OUTPUT_DTYPE = "bfloat16"

THREAD_BLOCK_M = 16
THREAD_BLOCK_N = 16
THREADS_PER_CTA = THREAD_BLOCK_M * THREAD_BLOCK_N
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 100


def _ensure_cuda_bindings_driver():
    """Provide the ``cuda.bindings.driver`` import expected by CuTeDSL."""
    try:
        import cuda.bindings.driver as cuda_driver  # type: ignore
    except ModuleNotFoundError:
        try:
            from cuda import cuda as legacy_cuda_driver  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "kernel_1_1_naive requires the CuTeDSL runtime, including "
                "nvidia-cutlass-dsl and CUDA Python bindings."
            ) from exc

        bindings_module = sys.modules.get("cuda.bindings")
        if bindings_module is None:
            bindings_module = types.ModuleType("cuda.bindings")
            bindings_module.__path__ = []
            sys.modules["cuda.bindings"] = bindings_module
        bindings_module.driver = legacy_cuda_driver
        sys.modules["cuda.bindings.driver"] = legacy_cuda_driver
        return legacy_cuda_driver
    else:
        return cuda_driver


@lru_cache(maxsize=1)
def _get_cutedsl_launcher():
    """Construct the CuTeDSL kernel and its host launch wrapper once."""
    from cuda.bindings import driver as cuda

    import cutlass
    import cutlass.cute as cute

    io_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32

    @cute.kernel
    def kernel(
        mA_mk: cute.Tensor,
        mB_kn: cute.Tensor,
        mC_mn: cute.Tensor,
    ):
        thread_n, thread_m, _ = cute.arch.thread_idx()
        block_n, block_m, _ = cute.arch.block_idx()

        row = block_m * THREAD_BLOCK_M + thread_m
        column = block_n * THREAD_BLOCK_N + thread_n
        rows = cute.size(mC_mn, mode=[0])
        columns = cute.size(mC_mn, mode=[1])
        reduction_size = cute.size(mA_mk, mode=[1])

        if row < rows and column < columns:
            accumulator = acc_dtype(0.0)
            for k in cutlass.range(reduction_size, unroll=1):
                a_value = acc_dtype(mA_mk[(row, k)])
                b_value = acc_dtype(mB_kn[(k, column)])
                accumulator += a_value * b_value

            mC_mn[(row, column)] = accumulator.to(io_dtype)

    @cute.jit
    def host_function(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_shape = (
            cute.ceil_div(c.layout.shape[1], THREAD_BLOCK_N),
            cute.ceil_div(c.layout.shape[0], THREAD_BLOCK_M),
            1,
        )
        kernel(a, b, c).launch(
            grid=grid_shape,
            block=(THREAD_BLOCK_N, THREAD_BLOCK_M, 1),
            stream=stream,
        )

    return host_function


def run_matmul_local(
    dim: int = DEFAULT_DIM,
    dtype: str = DEFAULT_DTYPE,
    profile_region=None,
) -> dict:
    """Execute and time the naive CuTeDSL matrix multiplication."""
    try:
        _ensure_cuda_bindings_driver()
        import torch
        import cutlass.cute as cute
        from cuda.bindings import driver as cu_driver
        from cutlass.cute.runtime import from_dlpack
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "kernel_1_1_naive requires the CuTeDSL runtime, including "
            "nvidia-cutlass-dsl and CUDA Python bindings."
        ) from exc

    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")

    requested_dtype_name = str(dtype).lower()
    if requested_dtype_name not in (DEFAULT_DTYPE, "float16", "float32"):
        raise ValueError(
            f"Unsupported dtype '{requested_dtype_name}'. "
            f"kernel_1_1_naive only supports {DEFAULT_DTYPE} inputs."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run kernel_1_1_naive.")

    device = torch.device("cuda")
    torch_stream = torch.cuda.current_stream(device)
    current_stream = cu_driver.CUstream(torch_stream.cuda_stream)
    torch.manual_seed(1111)

    def make_tensor(rows: int, columns: int) -> torch.Tensor:
        return (
            torch.empty((rows, columns), dtype=torch.int32, device=device)
            .random_(-2, 2)
            .to(dtype=torch.bfloat16)
        )

    a = make_tensor(dim, dim)
    b = make_tensor(dim, dim)
    c = torch.zeros((dim, dim), dtype=torch.bfloat16, device=device)

    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=1)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=1)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=1)
    )

    # Exclude CuTeDSL JIT compilation from warmup and timed measurements.
    host_function = cute.compile(
        _get_cutedsl_launcher(),
        a_tensor,
        b_tensor,
        c_tensor,
        current_stream,
    )

    for _ in range(WARMUP_ITERATIONS):
        host_function(a_tensor, b_tensor, c_tensor, current_stream)
    torch_stream.synchronize()

    timed_matmul_region = profile_region or nullcontext
    profiler_enabled = profile_region is not None
    timed_iterations = 1 if profiler_enabled else TIMED_ITERATIONS

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with timed_matmul_region():
        start_event.record(torch_stream)
        for _ in range(timed_iterations):
            host_function(a_tensor, b_tensor, c_tensor, current_stream)
        end_event.record(torch_stream)
        torch_stream.synchronize()

    total_elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_ms = total_elapsed_ms / timed_iterations
    elapsed_s = elapsed_ms / 1000.0
    flops = 2 * dim * dim * dim
    tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0.0

    return {
        "status": "success",
        "kernel": "kernel_1_1_naive",
        "implementation": "cutedsl_scalar_naive",
        "operation": "scalar_dot_product_per_output",
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "requested_dtype": requested_dtype_name,
        "dtype": DEFAULT_DTYPE,
        "accumulation_dtype": ACCUMULATION_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "thread_block_shape": [THREAD_BLOCK_M, THREAD_BLOCK_N],
        "threads_per_cta": THREADS_PER_CTA,
        "device": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "timing_method": "cuda_events_explicit_stream_average",
        "profiler_enabled": profiler_enabled,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": timed_iterations,
        "aggregation": "single_total_elapsed_divided_by_timed_iterations",
        "total_elapsed_ms": round(total_elapsed_ms, 3),
        "elapsed_ms": round(elapsed_ms, 3),
        "flops": flops,
        "tflops": round(tflops, 6),
        "result_checksum": float(c.float().sum().item()),
    }
