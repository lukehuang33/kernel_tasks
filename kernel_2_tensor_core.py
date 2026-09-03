"""Blackwell Tensor Core GEMM without shared-memory swizzling in CuTeDSL.
- BF16 inputs
- FP32 accumulation in tensor memory
- BF16 outputs
- CTA-group-one tcgen05 MMA
- block tile shape 64 x 256 x 64
- B stored in transposed form (N x K)
- unswizzled K-major shared-memory layouts for A and B
"""

from contextlib import nullcontext
from functools import lru_cache
import sys
import types

DEFAULT_DIM = 4096
DEFAULT_DTYPE = "bfloat16"
DTYPE_CHOICES = ("bfloat16",)
ACCUMULATION_DTYPE = "float32"
OUTPUT_DTYPE = "bfloat16"

TILE_M = 64
TILE_N = 256
TILE_K = 64
THREADS_PER_CTA = 128
AB_STAGES = 1
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 100


def _ensure_cuda_bindings_driver():
    try:
        import cuda.bindings.driver as cuda_driver  # type: ignore
    except ModuleNotFoundError:
        try:
            from cuda import cuda as legacy_cuda_driver  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "kernel_2_tensor_core requires the CuTeDSL runtime, including "
                "nvidia-cutlass-dsl and CUDA Python bindings."
            ) from exc

        bindings_module = sys.modules.get("cuda.bindings")
        if bindings_module is None:
            bindings_module = types.ModuleType("cuda.bindings")
            bindings_module.__path__ = []
            sys.modules["cuda.bindings"] = bindings_module
        bindings_module.driver = legacy_cuda_driver
        sys.modules["cuda.bindings.driver"] = legacy_cuda_driver
    else:
        return cuda_driver


@lru_cache(maxsize=1) # kernel construction + wrapper is done once per python process
def _get_cutedsl_launcher():
    from cuda.bindings import driver as cuda

    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05

    io_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    mma_inst_shape_mnk = (TILE_M, TILE_N, 16) # instruction shape
    mma_tiler_mnk = (TILE_M, TILE_N, TILE_K) # block/CTA tile shape

    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1] # store barrier for A/B TMA copies
        mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1] # store barrier for UMMA shared-memory consumption
        tmem_holding_buf: cutlass.Int32 # tensor memory allocator

    @cute.kernel
    def kernel(
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_m, block_n, _ = cute.arch.block_idx()
        mma_coord_mnk = (block_m, block_n, None)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor( # set smem layout for A
            element_type=io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor( # set smem layout for B
            element_type=io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier( # create tmem barrier for allocator
            barrier_id=1,
            num_threads=THREADS_PER_CTA,
        )
        tmem = utils.TmemAllocator( # create tmem allocator to communicate tmeme address to cta
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(512) # tensor memory allocator

        if warp_idx == 0:
            # get descriptors for A and B matrices
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        num_tma_copy_bytes = cute.size_in_bytes(
            io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
        ab_mbar_ptr = storage.ab_mbar_ptr.data_ptr() # tma load completion barrier
        mma_mbar_ptr = storage.mma_mbar_ptr.data_ptr() # umma completion barrier before reading accumulators
        if warp_idx == 0: # issue barriers using the first warp
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(ab_mbar_ptr, 1)
                cute.arch.mbarrier_init(mma_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # tiles for A, B, C
        gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(0)

        # partition gmem tiles according to expected layout
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # smem fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # create accumulator shape
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        # prepare tma copy for A and B
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        cta_tile_shape_mnk = (
            mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
            mma_tiler_mnk[1],
            mma_tiler_mnk[2],
        )
        
        # get tile shape for epilogue load out of tmem and into shared mem
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk,
            False,
            utils.LayoutEnum.from_tensor(mC_mnl),
            io_dtype,
        )
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        tCtAcc_epi = cute.flat_divide(
            tCtAcc[((None, None), 0, 0)],
            epi_tile,
        )

        # define shape for tmem to registers
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x8)
            if mma_tiler_mnk[0] == 64
            else tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            cutlass.Float32,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # move data back to gmem (in C)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        num_k_tiles = cute.size(gA, mode=[2])
        tma_phase = cutlass.Int32(0)
        mma_phase = cutlass.Int32(0)
        issue_warp = warp_idx == 0
        if issue_warp:
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        # main loop for matmul
        for k_tile_idx in cutlass.range(num_k_tiles):
            if issue_warp:
                # warp 0 announces tma bytes, issues tma copies
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        ab_mbar_ptr,
                        num_tma_copy_bytes,
                    )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_idx)],
                    tAsA[(None, 0)],
                    tma_bar_ptr=ab_mbar_ptr,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_idx)],
                    tBsB[(None, 0)],
                    tma_bar_ptr=ab_mbar_ptr,
                )

            # all threads wait until tma copy is complete
            cute.arch.mbarrier_wait(ab_mbar_ptr, tma_phase)
            tma_phase ^= 1

            # warp 0 issues umma work
            if issue_warp:
                num_k_blocks = cute.size(tCrA, mode=[2])
                # loop 4 times, each is 64*256*16
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                with cute.arch.elect_one():
                    tcgen05.commit(mma_mbar_ptr)

            # wait on umma copy
            cute.arch.mbarrier_wait(mma_mbar_ptr, mma_phase)
            mma_phase ^= 1

        # release lock so hardware allocator could make progress
        tmem.relinquish_alloc_permit()

        # epilogue + tensor mem cleanup
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
            tTR_gC_slice = tTR_gC[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)
            tTR_gC_slice.store(tTR_rAcc.load().to(io_dtype))

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def host_function(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # create gemm object
        op = tcgen05.MmaF16BF16Op(
            io_dtype,
            acc_dtype,
            mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        # Build the K-major operand layouts explicitly instead of using the
        # Blackwell helper's swizzle-selection heuristic.
        a_smem_shape = tiled_mma.partition_shape_A(
            cute.dice(mma_tiler_mnk, (1, None, 1))
        )
        b_smem_shape = tiled_mma.partition_shape_B(
            cute.dice(mma_tiler_mnk, (None, 1, 1))
        )
        a_smem_layout_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_INTER,
            a.element_type,
        )
        b_smem_layout_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_INTER,
            b.element_type,
        )
        a_smem_layout = tcgen05.tile_to_mma_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, AB_STAGES),
            order=(1, 2, 3),
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, AB_STAGES),
            order=(1, 2, 3),
        )
        a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

        # define tma atom for A and B
        copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            copy_op,
            a,
            a_smem_layout_one_stage,
            mma_tiler_mnk,
            tiled_mma,
        )
        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op,
            b,
            b_smem_layout_one_stage,
            mma_tiler_mnk,
            tiled_mma,
        )

        # define grid shape and call kernel
        grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
        kernel(
            tiled_mma,
            a_tma_atom,
            a_tma_tensor,
            b_tma_atom,
            b_tma_tensor,
            c,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid_shape,
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
        )

    return host_function


def run_matmul_local(
    dim: int = DEFAULT_DIM,
    dtype: str = DEFAULT_DTYPE,
    profile_region=None,
) -> dict:
    """Execute the CuTeDSL tensor-core GEMM on Blackwell."""
    try:
        _ensure_cuda_bindings_driver()
        import torch
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cuda.bindings import driver as cu_driver
        from cutlass.cute.runtime import from_dlpack
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "kernel_2_tensor_core requires the CuTeDSL runtime, including "
            "nvidia-cutlass-dsl and CUDA Python bindings."
        ) from exc

    dim = int(dim)
    requested_dtype_name = str(dtype).lower()
    if requested_dtype_name not in (DEFAULT_DTYPE, "float16", "float32"):
        raise ValueError(
            f"Unsupported dtype '{requested_dtype_name}'. "
            f"kernel_2_tensor_core only supports {DEFAULT_DTYPE} inputs."
        )

    if dim % TILE_M != 0 or dim % TILE_N != 0 or dim % TILE_K != 0:
        raise ValueError(
            f"dim={dim} must be divisible by ({TILE_M}, {TILE_N}, {TILE_K}) "
            "for kernel_2_tensor_core."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run kernel_2_tensor_core.")

    device = torch.device("cuda")
    torch_stream = torch.cuda.current_stream(device)
    current_stream = cu_driver.CUstream(torch_stream.cuda_stream)
    torch.manual_seed(1111)

    def make_tensor(rows: int, cols: int) -> torch.Tensor:
        return (
            torch.empty((rows, cols), dtype=torch.int32, device=device)
            .random_(-2, 2)
            .to(dtype=torch.bfloat16)
        )

    a = make_tensor(dim, dim)
    b = make_tensor(dim, dim)
    c = torch.zeros((dim, dim), dtype=torch.bfloat16, device=device)

    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=dim)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=dim)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=dim)
    )

    # Compile once to a fixed JIT executor before warmup/timing.
    # print("[kernel_2] before launcher construction", flush=True)
    host_function = cute.compile(
        _get_cutedsl_launcher(),
        a_tensor,
        b_tensor,
        c_tensor,
        current_stream,
    )
    # print("[kernel_2] after launcher construction", flush=True)

    # Compilation above pays the JIT cost; warmups flush lazy runtime costs.
    for _ in range(WARMUP_ITERATIONS):
        # print("[kernel_2] before warmup launch", flush=True)
        host_function(a_tensor, b_tensor, c_tensor, current_stream)
    # print("[kernel_2] before warmup sync", flush=True)
    torch_stream.synchronize()
    # print("[kernel_2] after warmup sync", flush=True)

    timed_matmul_region = profile_region or nullcontext
    profiler_enabled = profile_region is not None
    timed_iterations = 1 if profiler_enabled else TIMED_ITERATIONS

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with timed_matmul_region():
        # print("[kernel_2] before timed launch", flush=True)
        start_event.record(torch_stream)
        for _ in range(timed_iterations):
            host_function(a_tensor, b_tensor, c_tensor, current_stream)
        end_event.record(torch_stream)
        # print("[kernel_2] before timed sync", flush=True)
        torch_stream.synchronize()
        # print("[kernel_2] after timed sync", flush=True)

    # print("[kernel_2] before elapsed time read", flush=True)
    total_elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_ms = total_elapsed_ms / timed_iterations
    elapsed_s = elapsed_ms / 1000.0
    flops = 2 * dim * dim * dim
    tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0.0

    # print("[kernel_2] before checksum", flush=True)
    result_checksum = float(c.float().sum().item())
    # print("[kernel_2] after checksum", flush=True)

    # print("[kernel_2] returning result", flush=True)
    return {
        "status": "success",
        "kernel": "kernel_2_tensor_core",
        "implementation": "cutedsl_blackwell_tcgen05_no_smem_swizzle",
        "operation": "cute.gemm",
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "requested_dtype": requested_dtype_name,
        "dtype": DEFAULT_DTYPE,
        "accumulation_dtype": ACCUMULATION_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "tile_shape": [TILE_M, TILE_N, TILE_K],
        "mma_shape": [TILE_M, TILE_N, 16],
        "threads_per_cta": THREADS_PER_CTA,
        "cta_group": 1,
        "smem_swizzle": "none",
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
        "result_checksum": result_checksum,
        "cutlass_io_dtype": str(cutlass_torch.dtype(cutlass.BFloat16)),
    }
