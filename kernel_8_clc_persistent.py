#!/usr/bin/env python3
"""
Blackwell Tensor Core GEMM with CLC persistent scheduling and a TMEM accumulator ring.

This kernel extends kernel_7_writeout_buffer.py and follows the kernel 8 changes from
Modular's Blackwell matmul repo:
- persistent work assignment through the hardware CLC scheduler,
- a dedicated scheduler warp,
- two-stage CLC response buffering,
- four accumulator stages in TMEM.

The CTA group still uses two CTAs, TMA for A/B loads, tcgen05 MMA, and a double-buffered
writeout path for C.
"""

import argparse
import importlib
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM", "1")

import modal


DEFAULT_DIM = 4096
DEFAULT_SEED = 0
DEFAULT_DTYPE = "bfloat16"
ACCUMULATION_DTYPE = "float32"
OUTPUT_DTYPE = "bfloat16"
DEFAULT_DEVICE = "CUDA"
DEFAULT_OUT_DIR = Path("/tmp/matmul-results")

# 128 x 128 x 64 per-CTA A SMEM tile
# 128 x 64 x 64 per-CTA B SMEM tile because we use a 4-stage accumulator
# in tensor memory, and we have 8 AB_stages. So, with 512 KiB TMEM space,
# we cannot load the 128 x 128 x 64 tile for B.
SMEM_TILE_M = 128
SMEM_TILE_N = 64
CTA_TILE_M = 128
CTA_TILE_N = 128
TILE_K = 64
MMA_TILE_M = 256
MMA_TILE_N = 128
MMA_TILE_K = 16

EPILOGUE_WARPS = 4
SCHEDULER_WARP_ID = 4 # add CTC schedule warp ID
TMA_WARP_ID = 5
MMA_WARP_ID = 6
THREADS_PER_CTA = 7 * 32 # extra schedular warp for CLC.
CTA_GROUP_SIZE = 2
CLUSTER_SHAPE_MN = (2, 1)
AB_STAGES = 8 # 8 A/B stages instead of 6
# increase ACC_STAGES to 4 (TMEM_N // MMA_N) to pipeline MMA accumulations using CTC
ACC_STAGES = 4 
CLC_STAGES = 2 # CLC stages to pipeline CLC fetch
EPI_STAGES = 2 # pipeline epilogue + write out
OUTPUT_STAGE_N = 32
TMEM_COLUMNS = 512
ACC_STAGE_STRIDE_COLS = TMEM_COLUMNS // ACC_STAGES

# Calculate thread numbers
EPILOGUE_THREADS = EPILOGUE_WARPS * 32
SCHEDULER_THREADS = 32
TMA_THREADS = 32
MMA_THREADS = 32
CLUSTER_SIZE = CLUSTER_SHAPE_MN[0] * CLUSTER_SHAPE_MN[1]
ACC_CONSUMER_THREADS = CTA_GROUP_SIZE * EPILOGUE_THREADS
CLC_CONSUMER_THREADS = SCHEDULER_THREADS + CLUSTER_SIZE * (
    TMA_THREADS + MMA_THREADS + EPILOGUE_THREADS
)

WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 50
DEFAULT_GPU = "B200"

_CUTE_DSL_LAUNCHER: Any | None = None


def _ensure_cuda_bindings_driver() -> None:
    """Provide the cuda.bindings.driver alias expected by cutlass on Modal."""
    try:
        import cuda.bindings.driver  # type: ignore  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        cuda_module = importlib.import_module("cuda")
        cuda_driver_module = importlib.import_module("cuda.cuda")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on Modal image
        raise ModuleNotFoundError(
            "kernel8_clc_persistent requires cuda.bindings. Install cuda-bindings "
            "inside the Modal image."
        ) from exc

    bindings_module = sys.modules.get("cuda.bindings")
    if bindings_module is None:
        import types

        bindings_module = types.ModuleType("cuda.bindings")
        bindings_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["cuda.bindings"] = bindings_module
        setattr(cuda_module, "bindings", bindings_module)

    sys.modules["cuda.bindings.driver"] = cuda_driver_module
    setattr(bindings_module, "driver", cuda_driver_module)


def _get_cutedsl_launcher():
    """Build and cache the Cutedsl kernel launcher."""
    global _CUTE_DSL_LAUNCHER
    if _CUTE_DSL_LAUNCHER is not None:
        return _CUTE_DSL_LAUNCHER

    _ensure_cuda_bindings_driver()

    import cuda.bindings.driver as cuda_driver

    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.base_dsl.typing import Int128
    from cutlass.cute.nvgpu import cpasync, tcgen05

    io_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    mma_tiler_mnk = (MMA_TILE_M, MMA_TILE_N, TILE_K) # logical 2-CTA tile
    cta_tile_shape_mnk = (CTA_TILE_M, CTA_TILE_N, TILE_K)
    cluster_shape_mnl = (*CLUSTER_SHAPE_MN, 1)

    @cute.struct
    class SharedStorage:
        # PipelineTmaUmma uses a full and an empty mbarrier per circular stage.
        # The TMA and MMA warps independently advance through these stages.
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
        # Allocate memory for CLC memory barrier, CLC throttle barrier, CLC response
        clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
        clc_throttle_mbar_ptr: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
        clc_response_ptr: cute.struct.MemRange[Int128, CLC_STAGES]
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32

    @cute.kernel
    def _matmul_persistent_kernel(
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
        grid_dim = cute.arch.grid_dim()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor( # set smem layout for this CTA's half of A
            element_type=io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor( # set smem layout for this CTA's half of B
            element_type=io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )
        sC = smem.allocate_tensor( # two BM x 32 write-out buffers
            element_type=io_dtype,
            layout=epi_smem_layout.outer,
            byte_alignment=128,
            swizzle=epi_smem_layout.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier( # create tmem allocator barrier
            barrier_id=1,
            num_threads=THREADS_PER_CTA,
        )
        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=EPILOGUE_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # set clc layout
        clc_layout_vmnk = cute.make_layout(
            (CTA_GROUP_SIZE, 1, 1, 1),
            stride=(1, 1, 1, 1),
        )

        # A CTA pair owns one logical 256 x 256 output tile. The CTA's V
        # coordinate chooses the upper/lower 128-row TMEM and output slice.
        mma_tile_coord_v = block_m % CTA_GROUP_SIZE
        is_leader_cta = mma_tile_coord_v == 0 # make the first SM the leader
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        issue_warp = warp_idx == 0

        if warp_idx == TMA_WARP_ID:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Both CTAs issue their half-loads, but CTA-group-two TMA completes the
        # leader CTA's transaction barrier. It must therefore expect both CTAs'
        # A and B byte counts.
        num_tma_copy_bytes = (
            cute.size_in_bytes(
                io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
            )
            + cute.size_in_bytes(
                io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
            )
        ) * CTA_GROUP_SIZE

        # Cluster-aware full/empty barriers make the A/B buffers a circular
        # producer-consumer queue. The specialized TMA and MMA warps can advance
        # concurrently while still preventing either side from reusing a live
        # stage.
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tma_producers,
            ),
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            tx_count=num_tma_copy_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ACC_CONSUMER_THREADS
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_producer, acc_consumer = acc_pipeline.make_participants()

        # Make the pipeline for the CLC fetches to send next work tile info to warps. 
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, CLC_CONSUMER_THREADS
            ),
            tx_count=16,
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=clc_layout_vmnk,
        )
        clc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, CLC_STAGES
        )
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )

        # Make the pipeline for the CLC throttle. 
        clc_throttle_pipeline = pipeline.PipelineAsync.create(
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, TMA_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, SCHEDULER_THREADS
            ),
            barrier_storage=storage.clc_throttle_mbar_ptr.data_ptr(),
        )
        clc_throttle_producer, clc_throttle_consumer = (
            clc_throttle_pipeline.make_participants()
        )

        clc_response_base = storage.clc_response_ptr.data_ptr()
        cta_m_in_cluster = block_m % CLUSTER_SHAPE_MN[0]
        cta_n_in_cluster = block_n % CLUSTER_SHAPE_MN[1]
        cluster_dim_m = grid_dim[0] // CLUSTER_SHAPE_MN[0]

        tmem.allocate(TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)

        # Set up per-CTA view for smem A/B tiles
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        num_k_blocks = cute.size(tCrA, mode=[2])

        # Partition each operand by the cluster mode that shares it. CuTe uses
        # this partition together with the multicast masks below to address the
        # correct distributed-SMEM slice.
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
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
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_base = tiled_mma.make_fragment_C(acc_shape)
        tCtAcc_layout = tCtAcc_base.layout


        if warp_idx == TMA_WARP_ID:
            # Convert block coordinate into the work tile coordinate used by the scheduler.
            # Note that we use a simple snake swizzle.
            tma_cluster_m = block_m // CLUSTER_SHAPE_MN[0]
            tma_cluster_n = block_n // CLUSTER_SHAPE_MN[1]
            tma_swizzled_cluster_m = cutlass.Int32(
                cutlass.select_(
                    (tma_cluster_n % 2) == 0,
                    tma_cluster_m,
                    cluster_dim_m - tma_cluster_m - 1,
                )
            )

            # Get the work coordinates of the cta
            tma_work_m = (
                tma_swizzled_cluster_m * CLUSTER_SHAPE_MN[0] + cta_m_in_cluster
            )
            tma_work_n = tma_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
            tma_work_valid = cutlass.Boolean(True)
            while tma_work_valid:
                if is_first_cta_in_cluster:
                    clc_throttle_token = clc_throttle_producer.acquire_and_advance()
                    clc_throttle_token.commit()

                mma_coord_mnk = (tma_work_m // CTA_GROUP_SIZE, tma_work_n, None)

                # tiles for A, B in gmem
                gA = cute.local_tile(
                    mA_mkl,
                    mma_tiler_mnk,
                    mma_coord_mnk,
                    proj=(1, None, 1),
                )
                gB = cute.local_tile(
                    mB_nkl,
                    mma_tiler_mnk,
                    mma_coord_mnk,
                    proj=(None, 1, 1),
                )

                # partition tiles according to this CTA's half of the pair MMA
                tCgA = thr_mma.partition_A(gA)
                tCgB = thr_mma.partition_B(gB)

                # Partition each operand by the cluster mode that shares it. CuTe uses
                # this partition together with the multicast masks below to address the
                # correct distributed-SMEM slice.
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

                # TMA producer loop copying A/B tiles from gmem to smem
                num_k_tiles = cute.size(gA, mode=[2])
                for k_tile in cutlass.range(num_k_tiles, unroll=1):
                    ab_empty = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, k_tile)],
                        tAsA[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                        mcast_mask=a_multicast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, k_tile)],
                        tBsB[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                        mcast_mask=b_multicast_mask,
                    )

                # Consumer waits on tile to be loaded, then gets the response index.
                clc_pipeline.consumer_wait(clc_consumer_state)
                next_m, next_n, _, next_valid = cute.arch.clc_response(
                    clc_response_base + clc_consumer_state.index
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )

                # Convert block coordinate into the work tile coordinate used by the scheduler.
                # Note that we use a simple snake swizzle.
                tma_cluster_m = next_m // CLUSTER_SHAPE_MN[0]
                tma_cluster_n = next_n // CLUSTER_SHAPE_MN[1]
                tma_swizzled_cluster_m = cutlass.Int32(
                    cutlass.select_(
                        (tma_cluster_n % 2) == 0,
                        tma_cluster_m,
                        cluster_dim_m - tma_cluster_m - 1,
                    )
                )
                tma_work_m = (
                    tma_swizzled_cluster_m * CLUSTER_SHAPE_MN[0]
                    + cta_m_in_cluster
                )
                tma_work_n = tma_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
                tma_work_valid = cutlass.Boolean(next_valid)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            ab_producer.tail()

        if warp_idx == SCHEDULER_WARP_ID and is_first_cta_in_cluster:
            # get index of scheduler warp
            scheduler_cluster_m = block_m // CLUSTER_SHAPE_MN[0]
            scheduler_cluster_n = block_n // CLUSTER_SHAPE_MN[1]
            scheduler_swizzled_cluster_m = cutlass.Int32(
                cutlass.select_(
                    (scheduler_cluster_n % 2) == 0,
                    scheduler_cluster_m,
                    cluster_dim_m - scheduler_cluster_m - 1,
                )
            )
            scheduler_work_m = (
                scheduler_swizzled_cluster_m * CLUSTER_SHAPE_MN[0]
                + cta_m_in_cluster
            )
            scheduler_work_n = (
                scheduler_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
            )
            scheduler_work_valid = cutlass.Boolean(True)
            while scheduler_work_valid:
                # wait on throttle
                clc_throttle_token = clc_throttle_consumer.wait_and_advance()
                clc_throttle_token.release()

                # Issue one CLC query, place its async response into the CLC pipeline slot. 
                clc_pipeline.producer_acquire(clc_producer_state)
                clc_response_ptr = clc_response_base + clc_producer_state.index
                with cute.arch.elect_one():
                    cute.arch.issue_clc_query(
                        clc_pipeline.producer_get_barrier(clc_producer_state),
                        clc_response_ptr,
                    )
                clc_producer_state.advance()

                # Consumer side of CLC response pipeline. Waits for the scheduler's result,
                # reads it, makes the async shared memory write visible/ordered.
                clc_pipeline.consumer_wait(clc_consumer_state)
                next_m, next_n, _, next_valid = cute.arch.clc_response(
                    clc_response_base + clc_consumer_state.index
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                scheduler_cluster_m = next_m // CLUSTER_SHAPE_MN[0]
                scheduler_cluster_n = next_n // CLUSTER_SHAPE_MN[1]
                scheduler_swizzled_cluster_m = cutlass.Int32(
                    cutlass.select_(
                        (scheduler_cluster_n % 2) == 0,
                        scheduler_cluster_m,
                        cluster_dim_m - scheduler_cluster_m - 1,
                    )
                )
                scheduler_work_m = (
                    scheduler_swizzled_cluster_m * CLUSTER_SHAPE_MN[0]
                    + cta_m_in_cluster
                )
                scheduler_work_n = (
                    scheduler_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
                )
                scheduler_work_valid = cutlass.Boolean(next_valid)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            clc_pipeline.producer_tail(clc_producer_state)

        if warp_idx == MMA_WARP_ID:
            # Convert block coordinate into the work tile coordinate used by the MMA.
            # Note that we use a simple snake swizzle.
            mma_cluster_m = block_m // CLUSTER_SHAPE_MN[0]
            mma_cluster_n = block_n // CLUSTER_SHAPE_MN[1]
            mma_swizzled_cluster_m = cutlass.Int32(
                cutlass.select_(
                    (mma_cluster_n % 2) == 0,
                    mma_cluster_m,
                    cluster_dim_m - mma_cluster_m - 1,
                )
            )
            mma_work_m = (
                mma_swizzled_cluster_m * CLUSTER_SHAPE_MN[0] + cta_m_in_cluster
            )
            mma_work_n = mma_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
            mma_work_valid = cutlass.Boolean(True)
            while mma_work_valid:
                # Decode the next CLC response and apply the scheduler swizzle
                clc_pipeline.consumer_wait(clc_consumer_state)
                next_m, next_n, _, next_valid = cute.arch.clc_response(
                    clc_response_base + clc_consumer_state.index
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                next_cluster_m = next_m // CLUSTER_SHAPE_MN[0]
                next_cluster_n = next_n // CLUSTER_SHAPE_MN[1]
                next_swizzled_cluster_m = cutlass.Int32(
                    cutlass.select_(
                        (next_cluster_n % 2) == 0,
                        next_cluster_m,
                        cluster_dim_m - next_cluster_m - 1,
                    )
                )
                next_work_m = (
                    next_swizzled_cluster_m * CLUSTER_SHAPE_MN[0]
                    + cta_m_in_cluster
                )
                next_work_n = next_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
                next_work_valid = cutlass.Boolean(next_valid)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

                if is_leader_cta:
                    # If leader, reserve accumulator slot and create the tensor view 
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = cute.make_tensor(
                        tmem_ptr + acc_empty.index * ACC_STAGE_STRIDE_COLS,
                        tCtAcc_layout,
                    )

                    # Get gmem local tile
                    mma_coord_mnk = (mma_work_m // CTA_GROUP_SIZE, mma_work_n, None)
                    gA = cute.local_tile(
                        mA_mkl,
                        mma_tiler_mnk,
                        mma_coord_mnk,
                        proj=(1, None, 1),
                    )
                    gB = cute.local_tile(
                        mB_nkl,
                        mma_tiler_mnk,
                        mma_coord_mnk,
                        proj=(None, 1, 1),
                    )
                    tCgA = thr_mma.partition_A(gA)
                    tCgB = thr_mma.partition_B(gB)

                    # Do the accumulation
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    num_k_tiles = cute.size(gA, mode=[2])
                    for k_tile in cutlass.range(num_k_tiles, unroll=1):
                        ab_full = ab_consumer.wait_and_advance()
                        for k_blk_idx in cutlass.range(
                            num_k_blocks, unroll_full=True
                        ):
                            k_blk_coord = (
                                None,
                                None,
                                k_blk_idx,
                                ab_full.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[k_blk_coord],
                                tCrB[k_blk_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_full.release()

                    acc_empty.commit()
                # advance to next tile
                mma_work_m = next_work_m
                mma_work_n = next_work_n
                mma_work_valid = next_work_valid

        if warp_idx < EPILOGUE_WARPS:
            # Map this CTA's initial block coordinate to the scheduler's swizzled
            # CTA-level work tile coordinate.
            tmem.relinquish_alloc_permit()

            epi_cluster_m = block_m // CLUSTER_SHAPE_MN[0]
            epi_cluster_n = block_n // CLUSTER_SHAPE_MN[1]
            epi_swizzled_cluster_m = cutlass.Int32(
                cutlass.select_(
                    (epi_cluster_n % 2) == 0,
                    epi_cluster_m,
                    cluster_dim_m - epi_cluster_m - 1,
                )
            )
            epi_work_m = (
                epi_swizzled_cluster_m * CLUSTER_SHAPE_MN[0] + cta_m_in_cluster
            )
            epi_work_n = epi_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
            epi_work_valid = cutlass.Boolean(True)
            while epi_work_valid:
                # Epilogue reads one completed accumulator tile from TMEM, 
                # stores it to the right C tile.
                acc_full = acc_consumer.wait_and_advance()

                tCtAcc = cute.make_tensor(
                    tmem_ptr + acc_full.index * ACC_STAGE_STRIDE_COLS,
                    tCtAcc_layout,
                )
                mma_coord_mnk = (epi_work_m // CTA_GROUP_SIZE, epi_work_n, None)
                gC = cute.local_tile(
                    mC_mnl,
                    mma_tiler_mnk,
                    mma_coord_mnk,
                    proj=(1, 1, None),
                )
                # Partition C according to 2-CTA MMA.
                tCgC = thr_mma.partition_C(gC)
                # Split C and accumulator 128x128 into 128x32 chunks.
                tCgC_epi_stage = cute.flat_divide(
                    tCgC[((None, None), 0, 0)],
                    epi_tile,
                )
                tCtAcc_epi_stage = cute.flat_divide(
                    tCtAcc[((None, None), 0, 0)],
                    epi_tile,
                )

                # copy accumulator into C.
                copy_atom_t2r_stage = sm100_utils.get_tmem_load_op(
                    cta_tile_shape_mnk,
                    c_smem_layout_kind,
                    io_dtype,
                    acc_dtype,
                    epi_tile,
                    True,
                )

                # Prep tile for copying into SMEM.
                tiled_copy_t2r_stage = tcgen05.make_tmem_copy(
                    copy_atom_t2r_stage, tCtAcc_epi_stage[(None, None, 0, 0)]
                )
                thr_copy_t2r_stage = tiled_copy_t2r_stage.get_slice(tidx)
                tTR_tAcc_stage = thr_copy_t2r_stage.partition_S(tCtAcc_epi_stage)
                tTR_gC_stage = thr_copy_t2r_stage.partition_D(tCgC_epi_stage)
                tTR_rAcc_stage = cute.make_rmem_tensor(
                    tTR_gC_stage[(None, None, None, 0, 0)].shape, acc_dtype
                )
                tTR_tAcc_stage = cute.group_modes(
                    tTR_tAcc_stage, 3, cute.rank(tTR_tAcc_stage)
                )

                # pack output in shared memory with stmatrix before TMA store
                copy_atom_r2s_stage = sm100_utils.get_smem_store_op(
                    c_smem_layout_kind,
                    io_dtype,
                    acc_dtype,
                    tiled_copy_t2r_stage,
                )
                # Set up the epilogue copy from registers to shared memory.
                tiled_copy_r2s_stage = cute.make_tiled_copy_D(
                    copy_atom_r2s_stage,
                    tiled_copy_t2r_stage,
                )
                thr_copy_r2s_stage = tiled_copy_r2s_stage.get_slice(tidx)
                tRS_sC_stage = thr_copy_r2s_stage.partition_D(sC)
                tRS_rAcc_stage = tiled_copy_r2s_stage.retile(tTR_rAcc_stage)
                tRS_rC_stage = cute.make_rmem_tensor(tRS_rAcc_stage.shape, io_dtype)

                # Set up the epilogue copy from shared memory to global memory
                tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(tCgC_epi_stage, 0, 2),
                )
                tCgC_grouped = cute.group_modes(tCgC_tma, 1, cute.rank(tCgC_tma))

                subtile_cnt = cute.size(tTR_tAcc_stage.shape, mode=[3])
                last_subtile_idx = subtile_cnt - 1
                # matmul tile loop
                for subtile_idx in cutlass.range(subtile_cnt):
                    # TMEM accumulator load for one epilogue subtile
                    tTR_tAcc_slice = tTR_tAcc_stage[
                        (None, None, None, subtile_idx)
                    ]
                    cute.copy(tiled_copy_t2r_stage, tTR_tAcc_slice, tTR_rAcc_stage)

                    # Register to SMEM epilogue step for one output subtile
                    c_buffer = subtile_idx % EPI_STAGES
                    tRS_sC_slice = tRS_sC_stage[(None, None, None, c_buffer)]

                    tRS_rC_stage.store(tRS_rAcc_stage.load().to(io_dtype))
                    cute.copy(tiled_copy_r2s_stage, tRS_rC_stage, tRS_sC_slice)

                    cute.arch.fence_view_async_shared()
                    epilogue_sync_barrier.arrive_and_wait()

                    # If we can issue the store, then copy back into GMEM
                    if issue_warp:
                        cute.copy(
                            tma_atom_c,
                            tCsC[(None, c_buffer)],
                            tCgC_grouped[(None, subtile_idx)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        if subtile_idx < last_subtile_idx:
                            cute.arch.cp_async_bulk_wait_group(
                                EPI_STAGES - 1,
                                read=True,
                            )
                        else:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)

                    if subtile_idx > 0:
                        if subtile_idx < last_subtile_idx:
                            epilogue_sync_barrier.arrive_and_wait()

                acc_full.release()

                # Wait for CLC query to complete, read next work coord
                clc_pipeline.consumer_wait(clc_consumer_state)
                next_m, next_n, _, next_valid = cute.arch.clc_response(
                    clc_response_base + clc_consumer_state.index
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )

                # Assign next C tile for epilogue

                epi_cluster_m = next_m // CLUSTER_SHAPE_MN[0]
                epi_cluster_n = next_n // CLUSTER_SHAPE_MN[1]
                epi_swizzled_cluster_m = cutlass.Int32(
                    cutlass.select_(
                        (epi_cluster_n % 2) == 0,
                        epi_cluster_m,
                        cluster_dim_m - epi_cluster_m - 1,
                    )
                )
                epi_work_m = (
                    epi_swizzled_cluster_m * CLUSTER_SHAPE_MN[0]
                    + cta_m_in_cluster
                )
                epi_work_n = epi_cluster_n * CLUSTER_SHAPE_MN[1] + cta_n_in_cluster
                epi_work_valid = cutlass.Boolean(next_valid)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def _launch(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda_driver.CUstream,
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

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a.element_type,
            AB_STAGES,
        )
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

        c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)
        # Kernel 7 fixes the output stage to BM x 32 and double-buffers those
        # SMEM tiles, matching 7_double_buf_writeout.mojo.
        epi_tile = (CTA_TILE_M, OUTPUT_STAGE_N)
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

        problem_shape_ntile_mnl = (
            cute.ceil_div(c.shape[0], cta_tile_shape_mnk[0]),
            cute.ceil_div(c.shape[1], cta_tile_shape_mnk[1]),
            1,
        )
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            cluster_shape_mnl,
        )
        grid = tile_sched_params.get_grid_shape()

        _matmul_persistent_kernel(
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
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=cluster_shape_mnl,
            stream=stream,
            min_blocks_per_mp=1,
        )

    _CUTE_DSL_LAUNCHER = _launch
    return _launch


def run_matmul_local(
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = TIMED_ITERATIONS,
    compile_only: bool = False,
    validate: bool = True,
    profile_region=None,
) -> dict[str, Any]:
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
            "kernel8_clc_persistent requires the CuTeDSL runtime, "
            "including nvidia-cutlass-dsl and CUDA Python bindings."
        ) from exc

    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim % MMA_TILE_M != 0 or dim % MMA_TILE_N != 0 or dim % TILE_K != 0:
        raise ValueError(
            f"dim={dim} must be divisible by "
            f"({MMA_TILE_M}, {MMA_TILE_N}, {TILE_K}) for "
            "kernel8_clc_persistent."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the remote worker.")

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run kernel8_clc_persistent.")

    device = torch.device("cuda")
    torch_stream = torch.cuda.current_stream(device)
    current_stream = cu_driver.CUstream(torch_stream.cuda_stream)
    torch.manual_seed(seed)

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

    compile_start = time.perf_counter()
    host_function = cute.compile(
        _get_cutedsl_launcher(),
        a_tensor,
        b_tensor,
        c_tensor,
        current_stream,
    )
    compile_elapsed = time.perf_counter() - compile_start

    result: dict[str, Any] = {
        "status": "compiled" if compile_only else "success",
        "kernel": "kernel8_clc_persistent",
        "implementation": "cutedsl_blackwell_clc_persistent_tmem_ping",
        "operation": "cute.gemm",
        "dim": dim,
        "seed": seed,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_c": list(c.shape),
        "requested_dtype": DEFAULT_DTYPE,
        "dtype": DEFAULT_DTYPE,
        "accumulation_dtype": ACCUMULATION_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "smem_tile_shape": [SMEM_TILE_M, SMEM_TILE_N, TILE_K],
        "cta_output_tile_shape": [CTA_TILE_M, CTA_TILE_N],
        "pair_tile_shape": [MMA_TILE_M, MMA_TILE_N, TILE_K],
        "mma_shape": [MMA_TILE_M, MMA_TILE_N, MMA_TILE_K],
        "threads_per_cta": THREADS_PER_CTA,
        "epilogue_warps": EPILOGUE_WARPS,
        "scheduler_warp_id": SCHEDULER_WARP_ID,
        "tma_warp_id": TMA_WARP_ID,
        "mma_warp_id": MMA_WARP_ID,
        "cta_group": CTA_GROUP_SIZE,
        "cluster_shape": list(CLUSTER_SHAPE_MN),
        "ab_stages": AB_STAGES,
        "accumulator_stages": ACC_STAGES,
        "clc_stages": CLC_STAGES,
        "epilogue_stages": EPI_STAGES,
        "epilogue_tile_shape": [CTA_TILE_M, OUTPUT_STAGE_N],
        "tmem_columns": TMEM_COLUMNS,
        "tmem_accumulator_stage_stride_cols": ACC_STAGE_STRIDE_COLS,
        "device": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "compile_seconds": compile_elapsed,
        "compiled": True,
    }

    if compile_only:
        return result

    for _ in range(warmup):
        host_function(a_tensor, b_tensor, c_tensor, current_stream)
    torch_stream.synchronize()

    timed_matmul_region = profile_region or nullcontext
    profiler_enabled = profile_region is not None
    timed_iterations = 1 if profiler_enabled else int(iterations)

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

    result.update(
        {
            "timing_method": "cuda_events_explicit_stream_average",
            "profiler_enabled": profiler_enabled,
            "warmup_iterations": warmup,
            "timed_iterations": timed_iterations,
            "aggregation": "single_total_elapsed_divided_by_timed_iterations",
            "total_elapsed_ms": round(total_elapsed_ms, 3),
            "elapsed_ms": round(elapsed_ms, 3),
            "avg_ms": elapsed_ms,
            "flops": flops,
            "tflops": round(tflops, 6),
        }
    )

    if validate:
        expected = torch.matmul(a.float(), b.float().transpose(0, 1)).to(
            torch.bfloat16
        )
        torch_stream.synchronize()
        try:
            torch.testing.assert_close(
                c,
                expected,
                atol=2e-2,
                rtol=1e-2,
            )
            result["validated"] = True
        except AssertionError as exc:
            result["status"] = "validation_failed"
            result["validated"] = False
            result["validation_error"] = str(exc)
        max_abs_error = float((c.float() - expected.float()).abs().max().item())
        result["max_abs_error"] = max_abs_error
        result["max_abs_diff"] = max_abs_error
        result["correctness_check"] = "torch_float32_reference"

    result["result_checksum"] = float(c.float().sum().item())
    result["cutlass_io_dtype"] = str(cutlass_torch.dtype(cutlass.BFloat16))

    return result


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "cuda-bindings==12.9.0",
        "torch",
        "nvidia-cutlass-dsl==4.5.0.dev0",
        "numpy",
    )
    .add_local_file(__file__, remote_path="/root/kernel_8_clc_persistent.py")
)

app = modal.App("blackwell-matmul-kernel-8-clc-persistent", image=image)


@app.function(gpu=DEFAULT_GPU, timeout=60 * 20)
def run_remote(
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = TIMED_ITERATIONS,
    compile_only: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    return run_matmul_local(dim, seed, warmup, iterations, compile_only, validate)


def run_modal(
    dim: int,
    seed: int,
    warmup: int,
    iterations: int,
    compile_only: bool,
    validate: bool,
) -> dict[str, Any]:
    with app.run():
        return run_remote.remote(dim, seed, warmup, iterations, compile_only, validate)


def _format_result(result: dict[str, Any]) -> str:
    lines = [
        f"Kernel: {result.get('kernel')}",
        f"Dimension: {result.get('dim')} x {result.get('dim')}",
        f"Seed: {result.get('seed')}",
        f"Compiled: {result.get('compiled')} in {result.get('compile_seconds', 0):.3f}s",
    ]
    if "avg_ms" in result:
        lines.append(f"Average time: {result['avg_ms']:.4f} ms")
        lines.append(f"Throughput: {result['tflops']:.2f} TFLOP/s")
    if "validated" in result:
        lines.append(f"Validated: {result['validated']}")
        lines.append(f"Max abs diff: {result.get('max_abs_diff')}")
        if not result["validated"]:
            lines.append(result.get("validation_error", "validation failed"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Cutedsl kernel 8 Blackwell matmul benchmark"
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=TIMED_ITERATIONS)
    parser.add_argument("--local", action="store_true", help="Run in the current process")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile and launch once",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip comparison with torch.matmul",
    )
    args = parser.parse_args(argv)

    validate = not args.no_validate
    if args.local:
        result = run_matmul_local(
            dim=args.dim,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            compile_only=args.compile_only,
            validate=validate,
        )
    else:
        result = run_modal(
            dim=args.dim,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            compile_only=args.compile_only,
            validate=validate,
        )

    print(_format_result(result))
    return 0 if result.get("validated", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
