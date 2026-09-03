"""Blackwell Tensor Core GEMM with TMA multicast and 2-SM MMA.

Starts from kernel_4_stmatrix and adds the optimizations described in
Modular's kernel 5, written in CuTeDSL:

- BF16 inputs
- FP32 accumulation in tensor memory
- BF16 outputs
- CTA-group-two tcgen05 MMA across a 2 x 1 CTA cluster
- per-CTA A/B shared-memory tile shape 128 x 128 x 64
- logical pair-MMA tile shape 256 x 256 x 64
- TMA multicast/CTA-group-two loads into distributed shared memory
- B stored in transposed form (N x K)
- stmatrix packs BF16 output in shared memory before TMA store

This intentionally keeps a single A/B stage and a serial
load -> wait -> MMA -> wait schedule. Pipelining and warp specialization are
kernel 6 optimizations and do not belong in this kernel.
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

# Basing off of Modular's kernel 5, each CTA owns a 128 x 128 x 64 A/B SMEM tile. The two
# CTAs in a pair jointly issue a 256 x 256 x 16 MMA instruction, and each CTA
# owns a 128 x 256 slice of the output.
SMEM_TILE_M = 128
SMEM_TILE_N = 128
CTA_TILE_M = 128
CTA_TILE_N = 256
TILE_K = 64
MMA_TILE_M = 256
MMA_TILE_N = 256
MMA_TILE_K = 16
THREADS_PER_CTA = 128
CTA_GROUP_SIZE = 2
CLUSTER_SHAPE_MN = (2, 1)
AB_STAGES = 1
ACC_STAGES = 1
EPI_STAGES = 1

WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 50


def _ensure_cuda_bindings_driver():
    try:
        import cuda.bindings.driver as cuda_driver  # type: ignore
    except ModuleNotFoundError:
        try:
            from cuda import cuda as legacy_cuda_driver  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "kernel_5_multicast_2smmma requires the CuTeDSL runtime, "
                "including nvidia-cutlass-dsl and CUDA Python bindings."
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


@lru_cache(maxsize=1)  # kernel construction + wrapper is done once per python process
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
    mma_tiler_mnk = (MMA_TILE_M, MMA_TILE_N, TILE_K)  # logical 2-CTA tile
    cta_tile_shape_mnk = (CTA_TILE_M, CTA_TILE_N, TILE_K)
    cluster_shape_mnl = (*CLUSTER_SHAPE_MN, 1)

    @cute.struct
    class SharedStorage:
        # PipelineTmaUmma uses a full and an empty mbarrier per stage. There is
        # only one stage here, so this is synchronization rather than kernel-6
        # pipelining.
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
        tmem_dealloc_mbar_ptr: cutlass.Int64  # coordinates pair-CTA TMEM free
        tmem_holding_buf: cutlass.Int32  # tensor memory allocator result

    @cute.kernel
    def kernel(
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_kind: cutlass.Constexpr,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        num_tma_producers: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_m, block_n, _ = cute.arch.block_idx()

        # A CTA pair owns one logical 256 x 256 output tile. The CTA's V
        # coordinate chooses the upper/lower 128-row TMEM and output slice.
        mma_tile_coord_v = block_m % CTA_GROUP_SIZE
        is_leader_cta = mma_tile_coord_v == 0 # make the first SM the leader
        mma_coord_mnk = (block_m // CTA_GROUP_SIZE, block_n, None)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(  # set smem layout for this CTA's half of A
            element_type=io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(  # set smem layout for this CTA's half of B
            element_type=io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )
        sC = smem.allocate_tensor(  # kernel 4 epilogue SMEM is still used
            element_type=io_dtype,
            layout=epi_smem_layout.outer,
            byte_alignment=128,
            swizzle=epi_smem_layout.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(  # create tmem allocator barrier
            barrier_id=1,
            num_threads=THREADS_PER_CTA,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=THREADS_PER_CTA,
        )
        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=EPI_STAGES,
            producer_group=epilogue_pipeline_producer_group,
        )

        if warp_idx == 0:
            # get descriptors for A, B, and C matrices
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Both CTAs issue their half-loads, but CTA-group-two TMA completes the
        # leader CTA's transaction barrier. It must therefore expect both CTAs'
        # A and B byte counts, matching expected_bytes in Modular's kernel 5.
        num_tma_copy_bytes = (
            cute.size_in_bytes(
                io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
            )
            + cute.size_in_bytes(
                io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
            )
        ) * CTA_GROUP_SIZE

        # One-stage cluster-aware barriers express kernel 5's blocking
        # load/wait and final MMA/wait protocol. No stages overlap.
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tma_producers,
            ),
            tx_count=num_tma_copy_bytes,
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                THREADS_PER_CTA,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        # tiles for A, B, C
        gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # partition gmem tiles according to this CTA's half of the pair MMA
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # smem fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # create accumulator shape
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)
        tmem.allocate(utils.get_num_tmem_alloc_cols(tCtAcc))

        # Partition each operand by the cluster mode that shares it. CuTe uses
        # this partition together with the multicast masks below to address the
        # correct distributed-SMEM slice.
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # get tile shape for epilogue load out of tmem and into shared mem
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        tCtAcc_epi = cute.flat_divide(
            tCtAcc[((None, None), 0, 0)],
            epi_tile,
        )
        tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        # Pair MMA has a different TMEM lane/value layout than kernel 4's
        # single-CTA 64-row instruction. Let the SM100 helper select the
        # matching tcgen05.ld operation for the per-CTA output tile.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_smem_layout_kind,
            io_dtype,
            acc_dtype,
            epi_tile,
            True,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # move data back to gmem (in C)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0)].shape, acc_dtype
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        # pack output in shared memory with stmatrix before TMA store
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_smem_layout_kind,
            io_dtype,
            acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
        tCgC_grouped = cute.group_modes(tCgC_tma, 1, cute.rank(tCgC_tma))

        # create_tma_multicast_mask includes the pair/V mode and selects the
        # cluster CTAs that receive each operand's transaction.
        a_multicast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk,
            block_in_cluster_coord_vmnk,
            mcast_mode=2,
        )
        b_multicast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk,
            block_in_cluster_coord_vmnk,
            mcast_mode=1,
        )

        num_k_tiles = cute.size(gA, mode=[2])
        issue_warp = warp_idx == 0
        if issue_warp:
            # Reserve the sole accumulator stage before issuing any UMMA work.
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            # Main loop for matmul. This is intentionally blocking: each TMA
            # transfer completes before pair MMA, and pair MMA releases the
            # sole SMEM stage before the next transfer can start.
            for k_tile_idx in cutlass.range(num_k_tiles):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=a_multicast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=b_multicast_mask,
                )

                # Only the even/leader CTA issues tcgen05.mma.cta_group::2.
                if is_leader_cta:
                    ab_full = ab_consumer.wait_and_advance()
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    # loop 4 times, each is 256 x 256 x 16 across the CTA pair
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, ab_full.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # The release is tied to UMMA completion and makes the one
                    # A/B stage writable for the next serial iteration.
                    ab_full.release()

            if is_leader_cta:
                # Multicast MMA completion to both CTAs before either reads its
                # half of the accumulator from tensor memory.
                acc_empty.commit()

        acc_full = acc_consumer.wait_and_advance()

        # epilogue + tensor mem cleanup
        tmem.relinquish_alloc_permit()
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)

            c_buffer = subtile_idx % EPI_STAGES
            tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]

            # convert fp32 accumulator values to bf16 in tRS_rC
            tRS_rC.store(tRS_rAcc.load().to(io_dtype))
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)

            cute.arch.fence_view_async_shared()  # make TMA see the SMEM writes
            cute.arch.sync_threads()

            if issue_warp:
                cute.copy(
                    tma_atom_c,
                    tCsC[(None, c_buffer)],
                    tCgC_grouped[(None, subtile_idx)],
                )
                epilogue_pipeline.producer_commit()
                epilogue_pipeline.producer_acquire()
            cute.arch.sync_threads()

        epilogue_pipeline.producer_tail()
        acc_full.release()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def host_function(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Create a 2-CTA MMA object. The logical 256-row result is divided
        # evenly between the two CTAs in tiled_mma.thr_id.
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            acc_dtype,
            tcgen05.CtaGroup.TWO,
            mma_tiler_mnk[:2],
        )

        # create layout for a
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a.element_type,
            AB_STAGES,
        )
        # create layout for b
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b.element_type,
            AB_STAGES,
        )
        a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

        # Divide the physical 2 x 1 cluster by the pair-MMA V layout. The
        # resulting (V,M,N,K) layout drives TMA slicing, multicast masks, and
        # pair-CTA barrier signaling.
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(cluster_shape_mnl),
            (tiled_mma.thr_id.shape,),
        )
        num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        num_tma_producers = num_mcast_ctas_a + num_mcast_ctas_b - 1

        # Define cluster-aware TMA atoms for CTA-group-two loads. For this
        # 2 x 1 configuration CuTe selects the appropriate multicast/group-two
        # operation and slices each CTA's 128-row/column contribution.
        a_copy_op = sm100_utils.cluster_shape_to_tma_atom_A(
            CLUSTER_SHAPE_MN, tiled_mma.thr_id
        )
        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_copy_op,
            a,
            a_smem_layout_one_stage,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        b_copy_op = sm100_utils.cluster_shape_to_tma_atom_B(
            CLUSTER_SHAPE_MN, tiled_mma.thr_id
        )
        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            b_copy_op,
            b,
            b_smem_layout_one_stage,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        # add epilogue layouts for c
        c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk,
            True,
            c_smem_layout_kind,
            io_dtype,
        )
        epi_smem_layout = sm100_utils.make_smem_layout_epi(
            io_dtype,
            c_smem_layout_kind,
            epi_tile,
            EPI_STAGES,
        )
        epi_smem_layout_one_stage = cute.slice_(epi_smem_layout, (None, None, 0))
        c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout_one_stage,
            epi_tile,
        )

        # Each grid CTA owns a 128 x 256 result slice; round the grid to whole
        # 2 x 1 clusters so every pair-MMA has both participating CTAs.
        grid_shape = cute.round_up(
            cute.ceil_div((*c.layout.shape, 1), cta_tile_shape_mnk[:2]),
            cluster_shape_mnl,
        )
        kernel(
            tiled_mma,
            a_tma_atom,
            a_tma_tensor,
            b_tma_atom,
            b_tma_tensor,
            c_tma_atom,
            c_tma_tensor,
            a_smem_layout,
            b_smem_layout,
            c_smem_layout_kind,
            epi_smem_layout,
            epi_tile,
            cluster_layout_vmnk,
            num_tma_producers,
        ).launch(
            grid=grid_shape,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=cluster_shape_mnl,
            stream=stream,
        )

    return host_function


def run_matmul_local(
    dim: int = DEFAULT_DIM,
    dtype: str = DEFAULT_DTYPE,
    profile_region=None,
) -> dict:
    """Execute the CuTeDSL multicast pair-MMA GEMM on Blackwell."""
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
            "kernel_5_multicast_2smmma requires the CuTeDSL runtime, "
            "including nvidia-cutlass-dsl and CUDA Python bindings."
        ) from exc

    dim = int(dim)
    requested_dtype_name = str(dtype).lower()
    if requested_dtype_name not in (DEFAULT_DTYPE, "float16", "float32"):
        raise ValueError(
            f"Unsupported dtype '{requested_dtype_name}'. "
            f"kernel_5_multicast_2smmma only supports {DEFAULT_DTYPE} inputs."
        )

    if (
        dim % MMA_TILE_M != 0
        or dim % MMA_TILE_N != 0
        or dim % TILE_K != 0
    ):
        raise ValueError(
            f"dim={dim} must be divisible by "
            f"({MMA_TILE_M}, {MMA_TILE_N}, {TILE_K}) for "
            "kernel_5_multicast_2smmma."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run kernel_5_multicast_2smmma.")

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
    host_function = cute.compile(
        _get_cutedsl_launcher(),
        a_tensor,
        b_tensor,
        c_tensor,
        current_stream,
    )

    # Compilation above pays the JIT cost; warmups flush lazy runtime costs.
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

    # Validate after the timed region so the reference GEMM does not affect the
    # reported kernel latency. B is stored as N x K, hence the transpose here.
    reference = torch.matmul(a.float(), b.float().transpose(0, 1)).to(
        dtype=torch.bfloat16
    )
    torch.testing.assert_close(c, reference, atol=1e-2, rtol=1e-2)
    max_abs_error = float((c.float() - reference.float()).abs().max().item())

    result_checksum = float(c.float().sum().item())

    return {
        "status": "success",
        "kernel": "kernel_5_multicast_2smmma",
        "implementation": "cutedsl_blackwell_tma_multicast_2sm_mma_stmatrix",
        "operation": "cute.gemm",
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "requested_dtype": requested_dtype_name,
        "dtype": DEFAULT_DTYPE,
        "accumulation_dtype": ACCUMULATION_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "smem_tile_shape": [SMEM_TILE_M, SMEM_TILE_N, TILE_K],
        "cta_output_tile_shape": [CTA_TILE_M, CTA_TILE_N],
        "pair_tile_shape": [MMA_TILE_M, MMA_TILE_N, TILE_K],
        "mma_shape": [MMA_TILE_M, MMA_TILE_N, MMA_TILE_K],
        "threads_per_cta": THREADS_PER_CTA,
        "cta_group": CTA_GROUP_SIZE,
        "cluster_shape": list(CLUSTER_SHAPE_MN),
        "ab_stages": AB_STAGES,
        "schedule": "serial_tma_wait_pair_mma_wait",
        "epilogue": "stmatrix_to_smem_tma_store",
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
        "correctness_check": "torch_float32_reference",
        "max_abs_error": max_abs_error,
        "cutlass_io_dtype": str(cutlass_torch.dtype(cutlass.BFloat16)),
    }
