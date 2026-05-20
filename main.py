"""Local FastAPI entry point for Runpod matmul benchmarks."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from kernel_1_naive import DEFAULT_DIM, DEFAULT_DTYPE
from runpod_run import RUNPOD_GPU_NAME, run_matmul

app = FastAPI(
    title="Runpod Matmul Test",
    description="Dispatch PyTorch matmul benchmarks to a remote Runpod B200 worker.",
    version="0.1.0",
)


class MatmulRequest(BaseModel):
    """Request payload for the remote matmul worker."""

    dim: int = Field(default=DEFAULT_DIM, gt=0)
    dtype: Literal["bfloat16"] = DEFAULT_DTYPE


@app.get("/")
def home() -> dict:
    return {
        "message": "Runpod matmul test",
        "docs": "/docs",
        "target_gpu": RUNPOD_GPU_NAME,
        "endpoints": {
            "run_matmul": "/matmul/run",
        },
    }


@app.get("/ping")
def ping() -> dict:
    return {"status": "healthy"}


@app.post("/matmul/run")
async def matmul(request: MatmulRequest) -> dict:
    """Dispatch the benchmark to the remote Runpod worker."""
    return await run_matmul(dim=request.dim, dtype=request.dtype)
