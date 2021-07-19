#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
ReductionParams innerNormalizationHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    bool persistence_required,
    const int64_t max_persistent_buffer_size,
    size_t vectorize_factor) {
  // Set some targets for parallelization
  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max(
          (scheduler_utils::lastPow2((int64_t)n_tensor_inputs) >> 1),
          (int64_t)1));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache = 32 * 1024;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;
  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  int64_t min_red_elems_per_thread = std::max(
      l1_cache / (n_tensor_inputs * max_input_dtype_size * active_threads),
      (int64_t)1);

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 = n_elems * max_input_dtype_size * n_tensor_inputs <
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  // If it fits in l2, we just want to make sure each thread uses 32Bytes.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? (int64_t)32 / max_input_dtype_size : 32;

  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(num_elems_in_reduction, min_red_elems_per_thread), (int64_t)32);

  // Take the smaller
  const int64_t warp_size =
      std::min(warp_size_based_on_l1, warp_size_based_on_l2);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t max_threads_in_block = std::min(
      warp_size, ceilDiv(num_elems_in_reduction, min_red_elems_per_thread));

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, warp_size * min_red_elems_per_thread);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(
        n_elems, warp_size * std::max(target_unroll, min_red_elems_per_thread));
  } else {
    // Steal reduction elements from threads if it helps us get a wave of blocks
    min_red_elems_per_thread = std::min(
        min_red_elems_per_thread,
        ceilDiv(
            num_elems_in_reduction * num_outputs_for_reduction,
            warp_size * device_multiprocessor_count));
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll *
          std::max(target_unroll, min_red_elems_per_thread) <
      n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
    max_threads_in_block = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size
  const int64_t max_multi_reduction_factor = std::max(
      (persistence_required ? (scheduler_utils::register_file_size * 3) /
               (max_persistent_buffer_size * 4)
                            : std::numeric_limits<int64_t>::max()),
      (int64_t)1);

  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Blocks for outputs
  int64_t godim = 1;

  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for reduction
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = true;

  // Unroll amount
  int64_t unroll_factor = 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(num_elems_in_reduction, (int64_t)warp_size);
  // Put everything else in bdimy for now
  bdimy = std::min(
      std::max(max_threads_in_block / bdimx, (int64_t)1),
      max_multi_reduction_factor);

  int64_t remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
  int64_t remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

  // Adjust blocking and setup unrolling
  // Disable unrolling on iteration domain for now. TODO: Re-enable.
  if (remainder_in_reduction == 1 && !persistence_required) {
    // Small number of reduction elements, don't try to unroll the reduction dim
    unroll_reduction = false;
    // Try unrolling output dimension
    unroll_factor = std::min(target_unroll, remainder_in_output);
    remainder_in_output =
        ceilDiv(num_outputs_for_reduction, unroll_factor * bdimy);
  } else {
    // If we have reduction elements left, re-adjust the block dims
    bdimx = std::min(
        ceilDiv(num_elems_in_reduction, min_red_elems_per_thread),
        max_threads_in_block);

    // Don't exceed target.
    bdimy = std::min(
        std::max(max_threads_in_block / bdimx, (int64_t)1),
        max_multi_reduction_factor);
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

    remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
    unroll_factor = std::min(remainder_in_reduction, target_unroll);

    // Disable unrolling on iteration domain for now. TODO: Re-enable.
    if (unroll_factor == 1 && !persistence_required) {
      // If we can't unroll reduction dim, unroll output dim
      unroll_reduction = false;
      unroll_factor = std::min(remainder_in_output, target_unroll);
      remainder_in_output =
          ceilDiv(num_outputs_for_reduction, bdimy * unroll_factor);
      // remainder_in_reduction =
      //     ceilDiv(num_elems_in_reduction, bdimx * min_red_elems_per_thread);
      // Leave this commented for clang, still think it's important to have
      // though
    }
    //  else {
    // remainder_in_reduction = ceilDiv(
    //     num_elems_in_reduction,
    //     bdimx * std::max(unroll_factor, min_red_elems_per_thread));
    // Leave this commented for clang, still think it's important to have though
    // }
  }

  godim = remainder_in_output;
  // unroll_reduction = true;
  // unroll_factor = 8;
  // Persistence size from buffers
  int64_t batches_per_block = ceilDiv(
      num_elems_in_reduction,
      bdimx * (unroll_reduction ? unroll_factor : (int64_t)1));
  // round up to multiple of 8 or pow2 whichever smaller
  auto round_up_pow2 = scheduler_utils::lastPow2(batches_per_block);
  if (round_up_pow2 < batches_per_block) {
    round_up_pow2 *= 2;
  }

  constexpr int64_t kEight = 8; // clang tidy

  auto round_up_8 = batches_per_block % kEight == 0
      ? batches_per_block
      : batches_per_block + (kEight - batches_per_block % kEight);

  batches_per_block = std::min(round_up_8, round_up_pow2);

  while (persistence_required && unroll_factor <= max_unroll &&
         batches_per_block % 2 == 0) {
    batches_per_block /= 2;
    unroll_factor *= 2;
  }

  bool vectorize = false;

  if (vectorize_factor > 1 && unroll_factor > 1 && unroll_reduction) {
    vectorize = true;
    unroll_factor = std::min(
        scheduler_utils::lastPow2(unroll_factor), (int64_t)vectorize_factor);
  }

  ReductionParams rparams;
  rparams.fastest_dim = true;
  rparams.cross_block = true;
  rparams.cross_grid = false;
  rparams.multiple_reds_per_blk =
      bdimy > 1 || (!unroll_reduction && unroll_factor);
  rparams.loop_unroll = unroll_factor;
  rparams.vectorize = vectorize;
  rparams.reduction_unroll = unroll_reduction;
  rparams.batches_per_block = batches_per_block;
  rparams.persistent_kernel = persistence_required;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  rparams.split_grid_dim = godim > scheduler_utils::x_grid_limit;

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      persistence_required ? LaunchParams::UNINITIALIZED_VAL : bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  rparams.tag = persistence_required ? "Inner normalization heuristic.\n"
                                     : "Multi inner reduction (norm heuristic)";

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
ReductionParams OuterNormalizationHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    bool persistence_required,
    const int64_t max_persistent_buffer_size,
    size_t vectorize_factor) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;
  const int64_t l2_cache_size =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  const int64_t warp_size =
      n_elems * max_input_dtype_size * n_tensor_inputs < l2_cache_size
      ? (int64_t)32 / max_input_dtype_size
      : 32;

  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t max_threads_in_block = warp_size;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max(
          (scheduler_utils::lastPow2((int64_t)n_tensor_inputs) >> 1),
          (int64_t)1));

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, (int64_t)warp_size);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(target_blocks, target_unroll);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * max_threads_in_block < n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
    max_threads_in_block = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size

  const int64_t max_multi_reduction_factor = std::max(
      (persistence_required ? (scheduler_utils::register_file_size * 3) /
               (max_persistent_buffer_size * 4)
                            : std::numeric_limits<int64_t>::max()),
      (int64_t)1);

  // To get to target threads:
  // Prioritize
  // (1) x dim in iter domain
  // (2) unrolling in iter domain
  // (3) y in reduction domain
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions - need to flip unrolling to reduction
  // domain for this

  // Blocks for outputs
  // int64_t gdimx = 1; // unused at this time, comment for clang tidy

  // Threads for reduction
  int64_t bdimy = 1;
  // Threads for output
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = true;

  // Unroll amount
  int64_t unroll_factor = 1;

  int64_t remainder_in_reduction = num_elems_in_reduction;
  int64_t remainder_in_output = num_outputs_for_reduction;

  if (ceilDiv(num_outputs_for_reduction, warp_size) <
      device_multiprocessor_count) {
    // If we can't hit a full wave, reduce the warp_size to increase
    // the number of blocks.  The warp should be reduced at a minimum
    // to the granularity that an SM would pull a unique portion of a
    // cacheline from the memory system or else there is no
    // benefit from spreading the work to a different block.
    // This is dependent on the data size of elements.
    const int64_t cache_sector_bytes = 32;
    int64_t min_outputs_per_block =
        std::max(cache_sector_bytes / max_input_dtype_size, (int64_t)1);
    bdimx = std::min(
        std::min(
            std::max(
                ceilDiv(
                    num_outputs_for_reduction, device_multiprocessor_count) /
                    min_outputs_per_block,
                (int64_t)1),
            (int64_t)1) *
            min_outputs_per_block,
        max_multi_reduction_factor);
  } else {
    bdimx = std::min(
        max_threads_in_block,
        ceilDiv(num_outputs_for_reduction, target_blocks));
    bdimx = std::min(std::max(bdimx, warp_size), max_multi_reduction_factor);
  }

  bdimy = std::min(
      std::max(max_threads_in_block / bdimx, (int64_t)1),
      num_elems_in_reduction);

  // remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);
  // unused, but only commenting for clang-tidy
  remainder_in_reduction = ceilDiv(remainder_in_reduction, bdimy);

  if (num_outputs_for_reduction >=
      device_multiprocessor_count * max_threads_in_block) {
    // If we easily saturate the GPU, don't use block dim y and unroll output
    // dimension, this could be a more gentle transition starting earlier
    bdimx = std::min(max_threads_in_block, max_multi_reduction_factor);
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);

    bdimy = 1;
    remainder_in_reduction = num_elems_in_reduction;

    // Assume unroll in output, switch to remainder if cross grid
    // Don't unroll if we don't have 2 full waves
    //
    // For persistent kernels disable unrolling of iteration
    // TODO: Check if reduction schedulers actually benefit from this, only
    // considered persistent kernels which have difficulty inlining intermediate
    // values so register space can blow up.
    // Disable unrolling on iteration domain for now.
    unroll_factor = persistence_required
        ? 1
        : std::min(
              ceilDiv(remainder_in_output, device_multiprocessor_count * 2),
              target_unroll);

    if (unroll_factor > 1) {
      unroll_reduction = false;
    }

    if (unroll_factor == 1 && remainder_in_reduction > 1) {
      // Try unrolling in reduction dimension
      unroll_factor = std::min(remainder_in_reduction, unroll_factor);
      // remainder_in_reduction = ceilDiv(remainder_in_reduction,
      // unroll_factor); Unused, comment for clang tidy.
    }
    //  else {
    // remainder_in_output =
    //     ceilDiv(num_outputs_for_reduction, bdimx * unroll_factor);
    // unused, comment for clang tidy
    // }
  } else {
    // Not many output elements, try unrolling reduction dimension
    unroll_factor = std::min(max_unroll, remainder_in_reduction);
    if (unroll_factor > 1) {
      unroll_reduction = true;
    }
  }

  // Persistence size from buffers
  int64_t batches_per_block = 1;
  if (persistence_required) {
    batches_per_block = ceilDiv(
        num_elems_in_reduction,
        bdimy * (unroll_reduction ? unroll_factor : (int64_t)1));
    // round up to multiple of 8 or pow2 whichever smaller
  }

  auto round_up_pow2 = scheduler_utils::lastPow2(batches_per_block);
  if (round_up_pow2 < batches_per_block) {
    round_up_pow2 *= 2;
  }

  constexpr int64_t kEight = 8; // clang tidy

  auto round_up_8 = batches_per_block % kEight == 0
      ? batches_per_block
      : batches_per_block + (kEight - batches_per_block % kEight);

  batches_per_block = std::min(round_up_8, round_up_pow2);

  bool vectorize = false;

  if (vectorize_factor > 1 && unroll_factor > 1 && !unroll_reduction) {
    vectorize = true;
    unroll_factor = std::min(
        scheduler_utils::lastPow2(unroll_factor), (int64_t)vectorize_factor);
  }

  ReductionParams rparams;
  rparams.fastest_dim = false;
  rparams.cross_block = true;
  rparams.cross_grid = false;
  rparams.multiple_reds_per_blk =
      bdimx > 1 || (!unroll_reduction && unroll_factor);
  rparams.loop_unroll = unroll_factor;
  rparams.vectorize = vectorize;
  rparams.reduction_unroll = unroll_reduction;
  rparams.batches_per_block = batches_per_block;
  rparams.persistent_kernel = persistence_required;

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      persistence_required ? LaunchParams::UNINITIALIZED_VAL : bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  rparams.tag = persistence_required ? "Outer normalization heuristic.\n"
                                     : "Multi outer reduction (norm heuristic)";

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

} // namespace

ReductionParams NormalizationHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_tensor_inputs,
    size_t max_input_dtype_size,
    bool persistence_required,
    const int64_t max_persistent_buffer_size,
    size_t vectorize_factor) {
  if (fastest_dim_reduction) {
    return innerNormalizationHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        persistence_required,
        max_persistent_buffer_size,
        vectorize_factor);
  } else {
    return OuterNormalizationHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        persistence_required,
        max_persistent_buffer_size,
        vectorize_factor);
  }
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  FUSER_PERF_SCOPE("getNormalizationHeuristics");

  FusionGuard fg(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto first_red_tv = reduction_tvs[0];

  TORCH_INTERNAL_ASSERT(
      first_red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      first_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = first_red_tv->definition();

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          (red_expr->getExprType().value() == ExprType::ReductionOp ||
           red_expr->getExprType().value() == ExprType::WelfordOp),
      "TensorView doesn't have a reduction.");

  size_t max_dtype_size = 1;
  size_t n_tensor_inputs = 0;
  for (auto inp : fusion->inputs()) {
    if (inp->isA<TensorView>()) {
      max_dtype_size =
          std::max(max_dtype_size, dataTypeSize(inp->getDataType().value()));
      n_tensor_inputs++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      n_tensor_inputs > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);
  bool requires_persistence = !persistent_buffers.buffers.empty();

  auto properties =
      scheduler_utils::getProperties(fusion, runtime_info, first_red_tv);

  auto max_persistent_size =
      scheduler_utils::persistentBufferSize(fusion, runtime_info);

  auto vectorizable_inputs_outputs =
      scheduler_utils::getVectorizableInputsOutputs(first_red_tv);

  // Vectorize as much as we can
  size_t vectorize_factor = std::numeric_limits<size_t>::max();

  for (auto tv : vectorizable_inputs_outputs) {
    const auto tv_vectorize_factor = runtime_info.getVectorizableWidth(tv);
    vectorize_factor = std::min(vectorize_factor, tv_vectorize_factor);
  }

  if (vectorize_factor == std::numeric_limits<size_t>::max()) {
    vectorize_factor = 1;
  }

  return NormalizationHeuristic(
      properties.reduction_numel,
      properties.iteration_numel,
      properties.fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size,
      requires_persistence,
      max_persistent_size,
      vectorize_factor);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  FUSER_PERF_SCOPE("getNormalizationHeuristics");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);
  return getNormalizationHeuristics(fusion, runtime_info);
}

namespace {

void schedulePersistentNormalization(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("schedulePersistentNormalization");
  FusionGuard fg(fusion);
  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.

  // Cache inputs if unrolled
  auto cached_inputs =
      scheduler_utils::cacheInputs(fusion, rparams.loop_unroll > 1);

  // Cache and fork  outputs
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, rparams.loop_unroll > 1);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  TORCH_INTERNAL_ASSERT(reduction_tvs.size());
  auto reduction_tv = reduction_tvs[0];

  auto dim_analysis =
      scheduler_utils::canonicalDimReduction(fusion, reduction_tv);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  TORCH_INTERNAL_ASSERT(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  TensorView* reference_tv = scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");

  scheduler_utils::multiReductionInliner(
      fusion,
      rparams,
      reduction_tv,
      reference_tv,
      reduction_tvs,
      cached_inputs,
      cached_outputs);
}

void scheduleMultiReduction(Fusion* fusion, const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("scheduleMultiReduction");
  FusionGuard fg(fusion);
  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.

  // Cache inputs if unrolled
  auto cached_inputs =
      scheduler_utils::cacheInputs(fusion, rparams.loop_unroll > 1);

  // Cache and fork  outputs
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, rparams.loop_unroll > 1);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  TORCH_INTERNAL_ASSERT(reduction_tvs.size());
  auto reduction_tv = reduction_tvs[0];

  auto dim_analysis =
      scheduler_utils::canonicalDimReduction(fusion, reduction_tv);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  TORCH_INTERNAL_ASSERT(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  TensorView* reference_tv = scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");

  scheduler_utils::multiReductionInliner(
      fusion,
      rparams,
      reduction_tv,
      reference_tv,
      reduction_tvs,
      cached_inputs,
      cached_outputs);
}
} // namespace

// fusion is the input IR that will be modified by this function
TORCH_CUDA_CU_API void scheduleNormalization(
    Fusion* fusion,
    const ReductionParams& rparams) {
  if (rparams.persistent_kernel) {
    schedulePersistentNormalization(fusion, rparams);
  } else {
    scheduleMultiReduction(fusion, rparams);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch