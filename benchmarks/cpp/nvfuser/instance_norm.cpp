#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static void setupInstanceNorm(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  auto x = TensorViewBuilder().ndims(4).dtype(dtype).build();
  auto weight = TensorViewBuilder().ndims(1).dtype(dtype).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(dtype).build();
  auto running_mean =
      TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto running_var =
      TensorViewBuilder().ndims(1).dtype(DataType::Float).build();

  fusion->addInput(x);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  auto momentum_ptr = new Double(kMomentum);
  auto eps_ptr = new Double(kEps);

  auto norm = instance_norm(
      x,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr);
  auto norm_relu = unaryOp(UnaryOpType::Relu, norm.output);

  fusion->addOutput(norm_relu);
}

//------------------------------------------------------------------------------

static void nvFuserScheduler_InstanceNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};
  const auto aten_dtype = data_type_to_aten(dtype);

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_var = at::ones({input_shape[1]}, fp32_options);

  std::vector<c10::IValue> aten_inputs = {
      at_x, at_weight, at_bias, at_mean, at_var};
  std::vector<at::Tensor> outputs;

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
  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2) * dataTypeSize(DataType::Float)));
}

static void InstanceNorm_Baseline(
    benchmark::State& benchmark_state,
    DataType dtype) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const auto aten_dtype = data_type_to_aten(dtype);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto norm = at::instance_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);
    auto output = at::relu(norm);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2) * dataTypeSize(DataType::Float)));
}

//------------------------------------------------------------------------------

static void InstanceNorm_Baseline_fp32(benchmark::State& benchmark_state) {
  InstanceNorm_Baseline(benchmark_state, DataType::Float);
}

static void InstanceNorm_Baseline_fp16(benchmark::State& benchmark_state) {
  InstanceNorm_Baseline(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_fp32_InstanceNorm,
    setupInstanceNorm,
    nvFuserScheduler_InstanceNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp32_InstanceNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(InstanceNorm_Baseline_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(InstanceNorm_Baseline_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
