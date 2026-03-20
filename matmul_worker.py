"""Remote Runpod worker definitions for matmul benchmarks."""

from __future__ import annotations

from runpod_flash import GpuType, LiveServerless, remote

WORKER_NAME = "matmul-2048-b200"
DEFAULT_DIM = 4096
DEFAULT_DTYPE = "float32"
TARGET_GPU = GpuType.NVIDIA_B200

gpu_config = LiveServerless(
    name=WORKER_NAME,
    gpus=[TARGET_GPU],
    workersMin=0,
    workersMax=1,
    idleTimeout=5,
)


@remote(
    resource_config=gpu_config,
    dependencies=["torch"],
)
async def run_matmul(
    dim: int = 2048,
    dtype: str = "float32",
) -> dict:
    """Execute a vanilla PyTorch matmul on a remote B200 worker."""
    import time

    import torch

    dim = int(dim)
    dtype_name = str(dtype).lower()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Choose from: {', '.join(sorted(dtype_map))}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    device = torch.device("cuda")
    torch_dtype = dtype_map[dtype_name]

    a = torch.randn((dim, dim), device=device, dtype=torch_dtype)
    b = torch.randn((dim, dim), device=device, dtype=torch_dtype)

    # Warm up the CUDA context before timing the measured matmul.
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    started_at = time.perf_counter()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started_at
    elapsed_ms = elapsed_s * 1000.0

    # GEMM does 2 * M * N * K floating-point ops when counting mul+add separately.
    flops = 2 * dim * dim * dim
    tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0.0

    return {
        "status": "success",
        "operation": "torch.matmul",
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "elapsed_ms": round(elapsed_ms, 3),
        "flops": flops,
        "tflops": round(tflops, 6),
        "result_checksum": float(c.sum().item()),
    }
