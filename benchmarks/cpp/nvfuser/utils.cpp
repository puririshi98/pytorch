#include "utils.h"

#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <sstream>

using namespace torch::jit::fuser::cuda;

std::string toString(ReductionParams rparams) {
  std::stringstream ss;
  if (rparams.fastest_dim) {
    ss << "/Fastest dim";
  } else {
    ss << "/Slow dim";
  }
  if (rparams.cross_block) {
    ss << "/cross block";
  }
  if (rparams.multiple_reds_per_blk) {
    ss << "/multiple reductions per block ";
  }
  if (rparams.cross_grid) {
    ss << "/cross grid";
  }
  if (rparams.loop_unroll > 1) {
    ss << "/Unroll "
       << (rparams.reduction_unroll ? "reduction dim " : "iter dim ")
       << rparams.loop_unroll;
  }
  return ss.str();
}

std::string toString(LaunchParams lparams) {
  std::stringstream ss;
  lparams.toString();
  ss << "/Launch_Parameters["
     << "(" << lparams.bdimz() << "/" << lparams.bdimy() << "/"
     << lparams.bdimx() << ")/(" << lparams.gdimz() << "/" << lparams.gdimy()
     << "/" << lparams.gdimx() << ")/" << lparams.smem() << "]";
  return ss.str();
}

void clearL2Cache() {
  torch::NoGradGuard no_grad;
  auto l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA, 0);

  auto l2_elems = l2_cache_size / 4;
  torch::Tensor t0 = torch::empty(l2_elems, options);
  torch::Tensor t1 = torch::clone(t0);
};

void runBenchmarkIterations(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    std::vector<c10::IValue>& aten_inputs) {
  fusion_executor_cache->runFusionWithInputs(aten_inputs);
  bool segmented =
      fusion_executor_cache->getMostRecentKernelRuntime()->isSegmented();

  if (!segmented) {
    fusion_executor_cache->profile(true);
    fusion_executor_cache->runFusionWithInputs(aten_inputs);
    auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
    auto executor_instance = compile_log.fusion_executor;
    TORCH_INTERNAL_ASSERT(compile_log.reduction_params.has_value());
    TORCH_INTERNAL_ASSERT(compile_log.launch_constraints.has_value());
    auto rparams = toString(compile_log.reduction_params.value());
    auto lparams = toString(compile_log.launch_constraints.value());
    benchmark_state.SetLabel(rparams + lparams);
    executor_instance->setMeasureKernelTimeFlag(true);

    // Sync everything up before we start
    cudaDeviceSynchronize();
    for (auto _ : benchmark_state) {
      auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
      benchmark_state.SetIterationTime(
          executor_instance->kernelTimeMs() / 1000.0);
      clearL2Cache();
    }
    // Sync everything up before we're finished, don't want to run ahead on the
    // cpu while benchmarking.
    cudaDeviceSynchronize();
  } else {
    // Segmented
    // Sync everything up before we start
    cudaDeviceSynchronize();
    CudaKernelTimer timer;
    for (auto _ : benchmark_state) {
      timer.restart();
      auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
      benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
      clearL2Cache();
    }
    // Sync everything up before we're finished, don't want to run ahead on the
    // cpu while benchmarking.
    cudaDeviceSynchronize();
  }
}

namespace executorCache {
thread_local ExecutorMap executor_map_;
ExecutorMap& getGlobalMap() {
  return executor_map_;
}
} // namespace executorCache
