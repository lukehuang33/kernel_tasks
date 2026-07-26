"""Run the matmul benchmark remotely on Modal.

Usage:
    pip install modal
    modal setup
    modal run modal_run.py
    modal run modal_run.py --dim 8192
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import modal

REMOTE_PROJECT_DIR = "/root/runpod-matmul-test"
NSIGHT_VOLUME_NAME = "runpod-matmul-test-nsight"
NSIGHT_VOLUME_MOUNT = "/mnt/nsight"
if Path(REMOTE_PROJECT_DIR).exists() and REMOTE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, REMOTE_PROJECT_DIR)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DIM = 4096
DEFAULT_KERNEL = "kernel_2_tensor_core"
KERNEL_CHOICES = (
    "kernel_1_naive",
    "kernel_2_tensor_core",
    "kernel_4_stmatrix",
    "kernel_5_multicast_2smmma",
    "kernel_6_2sm_pipelining",
    "kernel_7_writeout_buffer",
    "kernel_8_clc_persistent",
    "kernel_9_thread_block_swizzle",
)
MODAL_GPU_NAME = "B200"
nsight_volume = modal.Volume.from_name(NSIGHT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.06-py3")
    .run_commands(
        "python -m pip uninstall -y "
        "cuda-python cuda-bindings "
        "nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base "
        "nvidia-cutlass-dsl-libs-cu13 || true"
    )
    .pip_install("cuda-bindings==12.9.0", "nvidia-cutlass-dsl==4.5.0.dev0")
    .add_local_file(
        str(PROJECT_DIR / "kernel_1_naive.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_1_naive.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_2_tensor_core.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_2_tensor_core.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_4_stmatrix.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_4_stmatrix.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_5_multicast_2smmma.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_5_multicast_2smmma.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_6_2sm_pipelining.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_6_2sm_pipelining.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_7_writeout_buffer.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_7_writeout_buffer.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_8_clc_persistent.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_8_clc_persistent.py",
    )
    .add_local_file(
        str(PROJECT_DIR / "kernel_9_thread_block_swizzle.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/kernel_9_thread_block_swizzle.py",
    )
)

app = modal.App("runpod-matmul-test")


def _load_kernel_module(kernel: str):
    kernel_name = str(kernel)
    if kernel_name == "kernel_1_naive":
        module_name = "kernel_1_naive"
    elif kernel_name == "kernel_2_tensor_core":
        module_name = "kernel_2_tensor_core"
    elif kernel_name == "kernel_4_stmatrix":
        module_name = "kernel_4_stmatrix"
    elif kernel_name == "kernel_5_multicast_2smmma":
        module_name = "kernel_5_multicast_2smmma"
    elif kernel_name == "kernel_6_2sm_pipelining":
        module_name = "kernel_6_2sm_pipelining"
    elif kernel_name == "kernel_7_writeout_buffer":
        module_name = "kernel_7_writeout_buffer"
    elif kernel_name == "kernel_8_clc_persistent":
        module_name = "kernel_8_clc_persistent"
    elif kernel_name == "kernel_9_thread_block_swizzle":
        module_name = "kernel_9_thread_block_swizzle"
    else:
        raise ValueError(
            f"Unsupported kernel '{kernel_name}'. Expected one of {KERNEL_CHOICES}."
        )
    return kernel_name, module_name, importlib.import_module(module_name)


@app.function(
    image=image,
    gpu=MODAL_GPU_NAME,
    timeout=600,
)
def run_matmul_modal(
    dim: int = DEFAULT_DIM,
    kernel: str = DEFAULT_KERNEL,
) -> dict:
    """Execute the shared matmul worker inside a Modal GPU container."""
    import os
    import sys

    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    kernel_name, module_name, module = _load_kernel_module(kernel)
    result = module.run_matmul_local(dim=dim)
    result["backend"] = "modal"
    result["requested_gpu"] = MODAL_GPU_NAME
    result["kernel"] = kernel_name
    result["kernel_module"] = module_name
    return result


@app.function(
    image=image,
    gpu=MODAL_GPU_NAME,
    timeout=1800,
    volumes={NSIGHT_VOLUME_MOUNT: nsight_volume},
)
def profile_matmul_modal(
    dim: int = DEFAULT_DIM,
    kernel: str = DEFAULT_KERNEL,
) -> dict:
    """Run the matmul worker under torch.profiler and persist the trace."""
    from contextlib import contextmanager
    import os

    import torch

    os.chdir(REMOTE_PROJECT_DIR)
    report_stem = (
        f"{NSIGHT_VOLUME_MOUNT}/{kernel}_{dim}_bfloat16_{int(time.time())}"
    )
    trace_filename = f"{Path(report_stem).name}.pt.trace.json"
    trace_path = Path(NSIGHT_VOLUME_MOUNT) / trace_filename

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )

    @contextmanager
    def profiled_matmul_region():
        profiler.start()
        try:
            with torch.profiler.record_function("timed_matmul"):
                yield
        finally:
            profiler.stop()

    kernel_name, module_name, module = _load_kernel_module(kernel)
    result = module.run_matmul_local(
        dim=dim,
        profile_region=profiled_matmul_region,
    )

    profiler.export_chrome_trace(str(trace_path))
    nsight_volume.commit()

    result["backend"] = "modal"
    result["requested_gpu"] = MODAL_GPU_NAME
    result["profiler"] = "torch.profiler"
    result["kernel"] = kernel_name
    result["kernel_module"] = module_name
    result["trace_volume"] = NSIGHT_VOLUME_NAME
    result["trace_filename"] = trace_filename
    result["trace_volume_path"] = f"/{trace_filename}"
    result["download_command"] = (
        f"modal volume get {NSIGHT_VOLUME_NAME} /{trace_filename} ./{trace_filename}"
    )
    return result


@app.local_entrypoint()
def main(
    dim: int = DEFAULT_DIM,
    kernel: str = DEFAULT_KERNEL,
) -> None:
    print(
        f"Dispatching {kernel} for {dim}x{dim} matrices on Modal {MODAL_GPU_NAME}..."
    )
    result = run_matmul_modal.remote(dim=dim, kernel=kernel)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def profile(
    dim: int = DEFAULT_DIM,
    kernel: str = DEFAULT_KERNEL,
) -> None:
    result = profile_matmul_modal.remote(dim=dim, kernel=kernel)
    print(json.dumps(result, indent=2, sort_keys=True))
