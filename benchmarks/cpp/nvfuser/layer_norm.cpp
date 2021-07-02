#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupLayerNorm(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  const int kReductionAxis = 1;
  const float kEps = 1e-5;

  Double* eps_ptr = new Double(kEps);

  // setup fusion
  auto input = TensorViewBuilder().ndims(2).dtype(dtype).build();
  fusion->addInput(input);
  auto layer_norm_results = layer_norm(input, 1, nullptr, nullptr, eps_ptr);
  fusion->addOutput(layer_norm_results.output);
}

static void nvFuserScheduler_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const float kEps = 1e-5;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({at_x});
  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs(aten_inputs);

  auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
  auto executor_instance = compile_log.fusion_executor;
  TORCH_INTERNAL_ASSERT(compile_log.reduction_params.has_value());
  TORCH_INTERNAL_ASSERT(compile_log.launch_constraints.has_value());
  auto rparams = toString(compile_log.reduction_params.value());
  auto lparams = toString(compile_log.launch_constraints.value());

  benchmark_state.SetLabel(rparams + lparams);

  fusion_executor_cache->profile(false);
  executor_instance->setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * 2 * at_x.numel() *
      int64_t(dataTypeSize(dtype)));
}

static void LayerNorm_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(at_x, norm_shape);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_fp32_LayerNorm,
    setupLayerNorm,
    nvFuserScheduler_LayerNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp32_LayerNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(LayerNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
