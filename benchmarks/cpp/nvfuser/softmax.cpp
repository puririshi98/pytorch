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
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);
  // setup fusion
  auto input = TensorViewBuilder().ndims(2).dtype(dtype).build();
  fusion->addInput(input);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
  }

  auto output = softmax(input, reduction_axis);

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

static void nvFuserScheduler_Softmax(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int reduction_axis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({aten_input});
  
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

//------------------------------------------------------------------------------

static void Softmax_Baseline(
    benchmark::State& benchmark_state,
    DataType dtype) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::_softmax(aten_input, kReductionAxis, false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

static void Softmax_Baseline_fp32(benchmark::State& benchmark_state) {
  Softmax_Baseline(benchmark_state, DataType::Float);
}

static void Softmax_Baseline_fp16(benchmark::State& benchmark_state) {
  Softmax_Baseline(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

static void setupSoftmaxDropout(
    Fusion* fusion,
    DataType dtype,
    const int kReductionAxis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

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

  if (dtype == DataType::Half) {
    attention_scores = castOp(DataType::Float, attention_scores);
    attention_mask = castOp(DataType::Float, attention_mask);
  }

  attention_scores = div(attention_scores, divisor);
  attention_scores = add(attention_scores, attention_mask);
  auto attention_probs = softmax(attention_scores, kReductionAxis);
  auto prob = new Double(kDropoutProbability);
  auto scale = new Double(kScale);
  auto dropout_results = dropout(attention_probs, prob, scale);
  auto output = dropout_results.output;

  if (dtype == DataType::Half) {
    attention_scores = castOp(DataType::Half, attention_scores);
    attention_probs = castOp(DataType::Half, attention_probs);
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(attention_scores);
  fusion->addOutput(attention_probs);
  fusion->addOutput(output);

  fusion->addOutput(dropout_results.mask);
}

static void nvFuserScheduler_SoftmaxDropout(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int kReductionAxis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  // reduce across 1, [256, 12, 100, 8]
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_scores = at::randn(input_shape, options);
  at::Tensor at_mask = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs(
      {at_scores, at_mask, sqrt(kAttentionHeadSize)});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

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
    const int kReductionAxis,
    DataType dtype) {
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr float kDropoutProbability = 0.1;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
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

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * 6 * 256 * 12 * 100 *
          benchmark_state.range(0) * int64_t(dataTypeSize(dtype)) +
      // bool mask
      int64_t(benchmark_state.iterations()) * 6 * 256 * 12 * 100 *
          benchmark_state.range(0) * int64_t(dataTypeSize(DataType::Bool)));
}

//------------------------------------------------------------------------------

static void Softmax_Dropout_Baseline_fp32_Inner(
    benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 3, DataType::Float);
}

static void Softmax_Dropout_Baseline_fp32_Outer(
    benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 1, DataType::Float);
}

static void Softmax_Dropout_Baseline_fp16_Inner(
    benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 3, DataType::Half);
}

static void Softmax_Dropout_Baseline_fp16_Outer(
    benchmark::State& benchmark_state) {
  Softmax_Dropout_Baseline(benchmark_state, 1, DataType::Half);
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
    nvFuserScheduler_fp16_Softmax_Outer,
    setupSoftmax,
    nvFuserScheduler_Softmax,
    DataType::Half,
    0);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp16_Softmax_Outer)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_fp16_Softmax_Inner,
    setupSoftmax,
    nvFuserScheduler_Softmax,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_fp16_Softmax_Inner)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_SoftmaxDropoutInner_fp32,
    setupSoftmaxDropout,
    nvFuserScheduler_SoftmaxDropout,
    DataType::Float,
    3);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutInner_fp32)
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
//     nvFuserScheduler_SoftmaxDropoutOuter_fp32,
//     setupSoftmaxDropout,
//     nvFuserScheduler_SoftmaxDropout,
//     DataType::Float,
//     1);

// NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutOuter_fp32)
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

NVFUSER_BENCHMARK_DEFINE(
    nvFuserScheduler_SoftmaxDropoutInner_fp16,
    setupSoftmaxDropout,
    nvFuserScheduler_SoftmaxDropout,
    DataType::Half,
    3);

NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutInner_fp16)
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
//     nvFuserScheduler_SoftmaxDropoutOuter_fp16,
//     setupSoftmaxDropout,
//     nvFuserScheduler_SoftmaxDropout,
//     DataType::Half,
//     1);

// NVFUSER_BENCHMARK_RUN(nvFuserScheduler_SoftmaxDropoutOuter_fp16)
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

BENCHMARK(Softmax_Baseline_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}, {0, 1}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Softmax_Baseline_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 12}, {0, 1}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Softmax_Dropout_Baseline_fp32_Inner)
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

BENCHMARK(Softmax_Dropout_Baseline_fp32_Outer)
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

BENCHMARK(Softmax_Dropout_Baseline_fp16_Inner)
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

BENCHMARK(Softmax_Dropout_Baseline_fp16_Outer)
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