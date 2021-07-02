#include <torch/csrc/jit/codegen/cuda/arith.h>
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

static void setupSoftmax(
    Fusion* fusion,
    DataType dtype,
    const int reduction_axis) {
  FusionGuard fg(fusion);
  // setup fusion
  auto input = TensorViewBuilder().ndims(2).dtype(dtype).build();
  fusion->addInput(input);
  auto output = softmax(input, reduction_axis);
  fusion->addOutput(output);
}

static void nvFuserScheduler_Softmax(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int reduction_axis) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs({aten_input});

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
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs({aten_input});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

static void Softmax_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::_softmax(at_x, kReductionAxis, false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

//------------------------------------------------------------------------------

static void setupSoftmaxDropout(
    Fusion* fusion,
    DataType dtype,
    const int kReductionAxis) {
  FusionGuard fg(fusion);

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  // setup fusion
  auto attention_scores = TensorViewBuilder()
                              .ndims(4)
                              .dtype(dtype)
                              .contiguity(std::vector<bool>(4, true))
                              .build();
  auto attention_mask = TensorViewBuilder()
                            .ndims(4)
                            .dtype(dtype)
                            .contiguity(std::vector<bool>(4, true))
                            .build();
  Double* divisor = new Double();
  fusion->addInput(attention_scores);
  fusion->addInput(attention_mask);
  fusion->addInput(divisor);

  attention_scores = div(attention_scores, divisor);
  attention_scores = add(attention_scores, attention_mask);
  auto attention_probs = softmax(attention_scores, kReductionAxis);
  auto prob = new Double(kDropoutProbability);
  auto scale = new Double(kScale);
  auto dropout_results = dropout(attention_probs, prob, scale);

  fusion->addOutput(attention_scores);
  fusion->addOutput(attention_probs);
  fusion->addOutput(dropout_results.output);
  fusion->addOutput(dropout_results.mask);
}

static void nvFuserScheduler_SoftmaxDropout(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int kReductionAxis) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // reduce across 1, [256, 12, 100, 8]
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_scores = at::randn(input_shape, options);
  at::Tensor at_mask = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs(
      {at_scores, at_mask, sqrt(kAttentionHeadSize)});

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
      int64_t(benchmark_state.iterations()) * 6 * 256 * 12 * 100 *
          benchmark_state.range(0) * int64_t(dataTypeSize(dtype)) +
      // bool mask
      int64_t(benchmark_state.iterations()) * 6 * 256 * 12 * 100 *
          benchmark_state.range(0) * int64_t(dataTypeSize(DataType::Bool)));
}

//------------------------------------------------------------------------------

static void Softmax_Dropout_Baseline(
    benchmark::State& benchmark_state,
    const int kReductionAxis) {
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr float kDropoutProbability = 0.1;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor attention_scores = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    // Create
    CudaKernelTimer timer;

    // Run
    attention_scores = attention_scores / sqrt(kAttentionHeadSize);
    attention_scores = attention_scores + at_y;
    auto attention_probs =
        at::_softmax(attention_scores, kReductionAxis, false);
    attention_probs = at::dropout(attention_probs, kDropoutProbability, true);

    // Record
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

//------------------------------------------------------------------------------

static void Softmax_Dropout_Baseline_Inner(benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 3);
}

static void Softmax_Dropout_Baseline_Outer(benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 1);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_fp32_Softmax_Outer,
    setupSoftmax,
    nvFuserScheduler_Softmax,
    DataType::Float,
    0);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp32_Softmax_Outer)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_fp32_Softmax_Inner,
    setupSoftmax,
    nvFuserScheduler_Softmax,
    DataType::Float,
    1);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp32_Softmax_Inner)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_SoftmaxDropoutInner,
    setupSoftmaxDropout,
    nvFuserScheduler_SoftmaxDropout,
    DataType::Float,
    3);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutInner)
    ->Arg(8)
    ->Arg(16)
    ->Arg(24)
    ->Arg(32)
    ->Arg(40)
    ->Arg(48)
    ->Arg(56)
    ->Arg(64)
    ->Arg(72)
    ->Arg(80)
    ->Arg(88)
    ->Arg(96)
    ->Arg(104)
    ->Arg(112)
    ->Arg(120)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TODO: Enable
// NVFUSER_BENCHMARK_DEFINE(
//     nvFuserScheduler_SoftmaxDropoutOuter,
//     setupSoftmaxDropout,
//     nvFuserScheduler_SoftmaxDropout,
//     DataType::Float,
//     1);

// NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutOuter)
//     ->Arg(8)
//     ->Arg(16)
//     ->Arg(24)
//     ->Arg(32)
//     ->Arg(40)
//     ->Arg(48)
//     ->Arg(56)
//     ->Arg(64)
//     ->Arg(72)
//     ->Arg(80)
//     ->Arg(88)
//     ->Arg(96)
//     ->Arg(104)
//     ->Arg(112)
//     ->Arg(120)
//     ->Arg(128)
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Softmax_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}, {0, 1}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Softmax_Dropout_Baseline_Inner)
    ->Arg(8)
    ->Arg(16)
    ->Arg(24)
    ->Arg(32)
    ->Arg(40)
    ->Arg(48)
    ->Arg(56)
    ->Arg(64)
    ->Arg(72)
    ->Arg(80)
    ->Arg(88)
    ->Arg(96)
    ->Arg(104)
    ->Arg(112)
    ->Arg(120)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Softmax_Dropout_Baseline_Outer)
    ->Arg(8)
    ->Arg(16)
    ->Arg(24)
    ->Arg(32)
    ->Arg(40)
    ->Arg(48)
    ->Arg(56)
    ->Arg(64)
    ->Arg(72)
    ->Arg(80)
    ->Arg(88)
    ->Arg(96)
    ->Arg(104)
    ->Arg(112)
    ->Arg(120)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
